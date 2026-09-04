//! Filesystem workspace tools (read_file, write_file, list_dir).

use super::security::safe_join;
use crate::schema::ToolResult;
use std::path::Path;

#[cfg(unix)]
use std::ffi::CString;
#[cfg(unix)]
use std::io::{self, Write};
#[cfg(unix)]
use std::os::fd::{AsRawFd, FromRawFd};
#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;
#[cfg(unix)]
use std::os::unix::fs::OpenOptionsExt;

#[inline]
pub(crate) fn get_str_arg<'a>(args: &'a serde_json::Value, key: &str) -> Result<&'a str, String> {
    args.get(key)
        .and_then(|v| v.as_str())
        .ok_or_else(|| format!("Missing required parameter '{}'", key))
}

fn with_safe_path<F>(
    tool_name: &'static str,
    args: &serde_json::Value,
    param: &str,
    ws: &Path,
    default_path: Option<&str>,
    op: F,
) -> ToolResult
where
    F: FnOnce(&Path, &str) -> std::result::Result<String, String>,
{
    let path_str = match default_path {
        Some(def) => args.get(param).and_then(|v| v.as_str()).unwrap_or(def),
        None => match get_str_arg(args, param) {
            Ok(p) => p,
            Err(e) => return ToolResult::err(tool_name, e),
        },
    };
    let target = match safe_join(ws, path_str) {
        Ok(t) => t,
        Err(e) => return ToolResult::err(tool_name, e),
    };
    match op(&target, path_str) {
        Ok(msg) => ToolResult::ok(tool_name, msg),
        Err(err) => ToolResult::err(tool_name, err),
    }
}

const MAX_FILE_READ_BYTES: u64 = 5 * 1024 * 1024; // 5 MB
const MAX_FILE_WRITE_BYTES: usize = 5 * 1024 * 1024; // 5 MB
const MAX_DIR_ENTRIES: usize = 500;

pub fn handle_read_file(args: serde_json::Value, ws: &Path) -> ToolResult {
    with_safe_path("read_file", &args, "path", ws, None, |target, path_str| {
        let meta = std::fs::metadata(target)
            .map_err(|e| format!("Failed to read metadata for '{}': {}", path_str, e))?;
        if !meta.is_file() {
            return Err(format!("'{}' is not a regular file", path_str));
        }
        if meta.len() > MAX_FILE_READ_BYTES {
            return Err(format!(
                "File '{}' size ({} bytes) exceeds maximum allowed read limit ({} bytes)",
                path_str,
                meta.len(),
                MAX_FILE_READ_BYTES
            ));
        }
        std::fs::read_to_string(target)
            .map_err(|e| format!("Failed to read file '{}': {}", path_str, e))
    })
}

pub fn handle_write_file(args: serde_json::Value, ws: &Path) -> ToolResult {
    let content = match get_str_arg(&args, "content") {
        Ok(c) => c,
        Err(e) => return ToolResult::err("write_file", e),
    };

    if content.len() > MAX_FILE_WRITE_BYTES {
        return ToolResult::err(
            "write_file",
            format!(
                "Content size ({} bytes) exceeds maximum write limit ({} bytes)",
                content.len(),
                MAX_FILE_WRITE_BYTES
            ),
        );
    }

    with_safe_path("write_file", &args, "path", ws, None, |_, path_str| {
        write_file_race_resistant(ws, path_str, content)
            .map(|_| {
                format!(
                    "Successfully wrote {} bytes to '{}'",
                    content.len(),
                    path_str
                )
            })
            .map_err(|e| format!("Failed to write file '{}': {}", path_str, e))
    })
}

/// Write a workspace file without following a path component that changes after
/// `safe_join` validates it. On Unix, every directory is opened relative to a
/// pinned parent descriptor with `O_NOFOLLOW`, and the replacement is performed
/// with `renameat` in that same directory. This keeps a local attacker from
/// swapping a directory or final symlink to redirect the write outside the
/// workspace between validation and the filesystem operation.
#[cfg(unix)]
fn write_file_race_resistant(workspace: &Path, relative_path: &str, content: &str) -> io::Result<()> {
    use std::fs::{File, OpenOptions};

    let components = Path::new(relative_path)
        .components()
        .filter_map(|component| match component {
            std::path::Component::Normal(name) => Some(name),
            std::path::Component::CurDir => None,
            _ => None,
        })
        .collect::<Vec<_>>();
    let file_name = components
        .last()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "empty workspace path"))?;

    let root = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(workspace)?;
    let mut parent = root;
    for component in &components[..components.len() - 1] {
        parent = open_or_create_directory_at(parent.as_raw_fd(), component)?;
    }

    let nonce = WRITE_TEMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let temporary_name = format!(
        ".mivi-write-{}-{}",
        std::process::id(),
        nonce
    );
    let temporary_name = CString::new(temporary_name).map_err(|_| {
        io::Error::new(io::ErrorKind::InvalidInput, "temporary filename contains NUL")
    })?;
    let file_name = CString::new(file_name.as_bytes()).map_err(|_| {
        io::Error::new(io::ErrorKind::InvalidInput, "filename contains NUL")
    })?;

    let temporary_fd = unsafe {
        // SAFETY: `parent` is an open directory descriptor owned by `parent`;
        // both C strings are NUL-free and remain alive for the syscall.
        libc::openat(
            parent.as_raw_fd(),
            temporary_name.as_ptr(),
            libc::O_WRONLY
                | libc::O_CREAT
                | libc::O_EXCL
                | libc::O_NOFOLLOW
                | libc::O_CLOEXEC,
            0o600,
        )
    };
    if temporary_fd < 0 {
        return Err(io::Error::last_os_error());
    }
    let mut temporary = unsafe {
        // SAFETY: `temporary_fd` is a newly opened, uniquely owned descriptor.
        File::from_raw_fd(temporary_fd)
    };

    let write_result = temporary.write_all(content.as_bytes()).and_then(|_| temporary.sync_all());
    if let Err(error) = write_result {
        let _ = unlink_at(parent.as_raw_fd(), temporary_name.as_ptr());
        return Err(error);
    }
    drop(temporary);

    let rename_result = unsafe {
        // SAFETY: both names are NUL-free and `parent` pins the directory used
        // for both operations; rename does not follow a final symlink.
        if libc::renameat(
            parent.as_raw_fd(),
            temporary_name.as_ptr(),
            parent.as_raw_fd(),
            file_name.as_ptr(),
        ) == 0
        {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    };
    if rename_result.is_err() {
        let _ = unlink_at(parent.as_raw_fd(), temporary_name.as_ptr());
    }
    rename_result
}

#[cfg(unix)]
fn open_or_create_directory_at(
    parent_fd: std::os::fd::RawFd,
    name: &std::ffi::OsStr,
) -> io::Result<std::fs::File> {
    let name = CString::new(name.as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "directory name contains NUL"))?;
    match open_directory_at(parent_fd, name.as_ptr()) {
        Ok(directory) => Ok(directory),
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            let created = unsafe {
                // SAFETY: `parent_fd` is an open directory descriptor and `name`
                // is a valid NUL-free component.
                libc::mkdirat(parent_fd, name.as_ptr(), 0o700)
            };
            if created < 0 {
                let create_error = io::Error::last_os_error();
                if create_error.raw_os_error() != Some(libc::EEXIST) {
                    return Err(create_error);
                }
            }
            open_directory_at(parent_fd, name.as_ptr())
        }
        Err(error) => Err(error),
    }
}

#[cfg(unix)]
fn open_directory_at(parent_fd: std::os::fd::RawFd, name: *const libc::c_char) -> io::Result<std::fs::File> {
    let fd = unsafe {
        // SAFETY: caller supplies a NUL-terminated component and `parent_fd`
        // is an open directory descriptor.
        libc::openat(
            parent_fd,
            name,
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        )
    };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(unsafe {
        // SAFETY: `fd` is newly opened and transferred to this File.
        std::fs::File::from_raw_fd(fd)
    })
}

#[cfg(unix)]
fn unlink_at(parent_fd: std::os::fd::RawFd, name: *const libc::c_char) -> io::Result<()> {
    let result = unsafe {
        // SAFETY: `parent_fd` is an open directory descriptor and `name` is a
        // valid NUL-terminated temporary filename.
        libc::unlinkat(parent_fd, name, 0)
    };
    if result == 0 {
        Ok(())
    } else {
        Err(io::Error::last_os_error())
    }
}

#[cfg(unix)]
static WRITE_TEMP_COUNTER: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

#[cfg(not(unix))]
fn write_file_race_resistant(workspace: &Path, relative_path: &str, content: &str) -> std::io::Result<()> {
    let target = workspace.join(relative_path);
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(target, content)
}

pub fn handle_list_dir(args: serde_json::Value, ws: &Path) -> ToolResult {
    with_safe_path(
        "list_dir",
        &args,
        "path",
        ws,
        Some("."),
        |target, path_str| {
            let entries = std::fs::read_dir(target)
                .map_err(|e| format!("Failed to list dir '{}': {}", path_str, e))?;
            let mut names = Vec::new();
            let mut truncated = false;
            for entry in entries.flatten() {
                if names.len() >= MAX_DIR_ENTRIES {
                    truncated = true;
                    break;
                }
                if let Ok(file_type) = entry.file_type() {
                    let kind = if file_type.is_dir() { "dir" } else { "file" };
                    let clean_name: String = entry
                        .file_name()
                        .to_string_lossy()
                        .chars()
                        .filter(|c| !c.is_control())
                        .collect();
                    names.push(format!("{}: {}", kind, clean_name));
                }
            }
            names.sort();
            if truncated {
                names.push(format!("... [truncated: showing first {} entries]", MAX_DIR_ENTRIES));
            }
            Ok(names.join("\n"))
        },
    )
}

#[cfg(all(test, unix))]
mod tests {
    use super::write_file_race_resistant;
    use std::fs;
    use std::os::unix::fs::symlink;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn race_resistant_write_rejects_a_final_symlink() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock must be after the Unix epoch")
            .as_nanos();
        let workspace = std::env::temp_dir().join(format!(
            "mivi-tools-write-{}-{}",
            std::process::id(),
            suffix
        ));
        let outside = PathBuf::from(format!("{}.outside", workspace.display()));
        fs::create_dir(&workspace).expect("create isolated workspace");
        fs::write(&outside, "keep me").expect("create outside file");
        symlink(&outside, workspace.join("target.txt")).expect("create test symlink");

        let result = write_file_race_resistant(&workspace, "target.txt", "overwrite me");

        assert!(result.is_ok(), "atomic replacement should not follow a final symlink");
        assert_eq!(fs::read_to_string(&outside).unwrap(), "keep me");
        assert_eq!(fs::read_to_string(workspace.join("target.txt")).unwrap(), "overwrite me");
        fs::remove_file(workspace.join("target.txt")).unwrap();
        fs::remove_file(outside).unwrap();
        fs::remove_dir(workspace).unwrap();
    }
}
