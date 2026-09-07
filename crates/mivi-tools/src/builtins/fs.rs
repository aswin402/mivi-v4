//! Filesystem workspace tools (read_file, write_file, list_dir).

use super::security::safe_join;
use crate::broker::ToolCancellation;
use crate::schema::ToolResult;
use std::path::Path;

#[cfg(unix)]
use std::ffi::CString;
#[cfg(unix)]
use std::io::{self, Write};
#[cfg(unix)]
use std::os::fd::{AsRawFd, FromRawFd, IntoRawFd};
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

const MAX_FILE_READ_BYTES: u64 = 5 * 1024 * 1024; // 5 MB
const MAX_FILE_WRITE_BYTES: usize = 5 * 1024 * 1024; // 5 MB
const MAX_DIR_ENTRIES: usize = 500;

pub fn handle_read_file(args: serde_json::Value, ws: &Path) -> ToolResult {
    handle_read_file_with_cancellation(args, ws, None)
}

pub fn handle_read_file_cancellable(
    args: serde_json::Value,
    ws: &Path,
    cancellation: &ToolCancellation,
) -> ToolResult {
    handle_read_file_with_cancellation(args, ws, Some(cancellation))
}

fn handle_read_file_with_cancellation(
    args: serde_json::Value,
    ws: &Path,
    cancellation: Option<&ToolCancellation>,
) -> ToolResult {
    let path_str = match get_str_arg(&args, "path") {
        Ok(path) => path,
        Err(error) => return ToolResult::err("read_file", error),
    };
    let result =
        read_workspace_file_with_cancellation(ws, path_str, MAX_FILE_READ_BYTES, cancellation);

    match result {
        Ok(content) => ToolResult::ok("read_file", content),
        Err(error) => ToolResult::err(
            "read_file",
            format!("Failed to read file '{}': {}", path_str, error),
        ),
    }
}

/// Read a UTF-8 workspace file without following a path component that can be
/// replaced after lexical validation. On Unix, the file and each parent
/// directory are opened relative to a pinned workspace descriptor with
/// `O_NOFOLLOW`. Other platforms use the validated path APIs available there.
pub fn read_workspace_file(
    workspace: &Path,
    relative_path: &str,
    max_bytes: u64,
) -> std::io::Result<String> {
    read_workspace_file_with_cancellation(workspace, relative_path, max_bytes, None)
}

fn read_workspace_file_with_cancellation(
    workspace: &Path,
    relative_path: &str,
    max_bytes: u64,
    cancellation: Option<&ToolCancellation>,
) -> std::io::Result<String> {
    check_cancellation(cancellation)?;
    let target = safe_join(workspace, relative_path)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidInput, error))?;

    #[cfg(unix)]
    {
        let _ = target;
        read_file_race_resistant(workspace, relative_path, max_bytes, cancellation)
    }
    #[cfg(not(unix))]
    {
        let metadata = std::fs::metadata(&target)?;
        if !metadata.is_file() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "target is not a regular file",
            ));
        }
        if metadata.len() > max_bytes {
            return Err(std::io::Error::new(
                std::io::ErrorKind::FileTooLarge,
                format!(
                    "file size ({} bytes) exceeds maximum allowed read limit ({})",
                    metadata.len(),
                    max_bytes
                ),
            ));
        }
        let file = std::fs::File::open(target)?;
        read_limited_file(file, max_bytes, cancellation)
    }
}

fn check_cancellation(cancellation: Option<&ToolCancellation>) -> std::io::Result<()> {
    if cancellation.is_some_and(ToolCancellation::is_cancelled) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::Interrupted,
            "tool execution cancelled",
        ));
    }
    Ok(())
}

fn read_limited_file<R: std::io::Read>(
    mut reader: R,
    max_bytes: u64,
    cancellation: Option<&ToolCancellation>,
) -> std::io::Result<String> {
    let mut bytes = Vec::new();
    let mut buffer = [0_u8; 16 * 1024];
    let read_limit = max_bytes.saturating_add(1);
    while (bytes.len() as u64) < read_limit {
        check_cancellation(cancellation)?;
        let remaining = read_limit - bytes.len() as u64;
        let read_size = remaining.min(buffer.len() as u64) as usize;
        let count = reader.read(&mut buffer[..read_size])?;
        if count == 0 {
            break;
        }
        bytes.extend_from_slice(&buffer[..count]);
    }
    check_cancellation(cancellation)?;
    if bytes.len() as u64 > max_bytes {
        return Err(std::io::Error::new(
            std::io::ErrorKind::FileTooLarge,
            format!(
                "file content exceeds maximum allowed read limit ({} bytes)",
                max_bytes
            ),
        ));
    }
    String::from_utf8(bytes)
        .map_err(|error| std::io::Error::new(std::io::ErrorKind::InvalidData, error.utf8_error()))
}

pub fn handle_write_file(args: serde_json::Value, ws: &Path) -> ToolResult {
    handle_write_file_with_cancellation(args, ws, None)
}

pub fn handle_write_file_cancellable(
    args: serde_json::Value,
    ws: &Path,
    cancellation: &ToolCancellation,
) -> ToolResult {
    handle_write_file_with_cancellation(args, ws, Some(cancellation))
}

fn handle_write_file_with_cancellation(
    args: serde_json::Value,
    ws: &Path,
    cancellation: Option<&ToolCancellation>,
) -> ToolResult {
    let content = match get_str_arg(&args, "content") {
        Ok(c) => c,
        Err(e) => return ToolResult::err("write_file", e),
    };

    if let Err(error) = check_cancellation(cancellation) {
        return ToolResult::err("write_file", error.to_string());
    }

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

    let path_str = match get_str_arg(&args, "path") {
        Ok(path) => path,
        Err(error) => return ToolResult::err("write_file", error),
    };
    if let Err(error) = safe_join(ws, path_str) {
        return ToolResult::err("write_file", error);
    }
    match write_file_race_resistant_with_cancellation(ws, path_str, content, cancellation) {
        Ok(()) => ToolResult::ok(
            "write_file",
            format!(
                "Successfully wrote {} bytes to '{}'",
                content.len(),
                path_str
            ),
        ),
        Err(error) => ToolResult::err(
            "write_file",
            format!("Failed to write file '{}': {}", path_str, error),
        ),
    }
}

/// Write a workspace file without following a path component that changes after
/// `safe_join` validates it. On Unix, every directory is opened relative to a
/// pinned parent descriptor with `O_NOFOLLOW`, and the replacement is performed
/// with `renameat` in that same directory. This keeps a local attacker from
/// swapping a directory or final symlink to redirect the write outside the
/// workspace between validation and the filesystem operation.
#[cfg(all(unix, test))]
fn write_file_race_resistant(
    workspace: &Path,
    relative_path: &str,
    content: &str,
) -> io::Result<()> {
    write_file_race_resistant_with_cancellation(workspace, relative_path, content, None)
}

#[cfg(unix)]
fn write_file_race_resistant_with_cancellation(
    workspace: &Path,
    relative_path: &str,
    content: &str,
    cancellation: Option<&ToolCancellation>,
) -> io::Result<()> {
    use std::fs::{File, OpenOptions};

    check_cancellation(cancellation)?;
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
        check_cancellation(cancellation)?;
        parent = open_or_create_directory_at(parent.as_raw_fd(), component)?;
    }

    let nonce = WRITE_TEMP_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let temporary_name = format!(".mivi-write-{}-{}", std::process::id(), nonce);
    let temporary_name = CString::new(temporary_name).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "temporary filename contains NUL",
        )
    })?;
    let file_name = CString::new(file_name.as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "filename contains NUL"))?;

    let temporary_fd = unsafe {
        // SAFETY: `parent` is an open directory descriptor owned by `parent`;
        // both C strings are NUL-free and remain alive for the syscall.
        libc::openat(
            parent.as_raw_fd(),
            temporary_name.as_ptr(),
            libc::O_WRONLY | libc::O_CREAT | libc::O_EXCL | libc::O_NOFOLLOW | libc::O_CLOEXEC,
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

    let write_result =
        write_all_with_cancellation(&mut temporary, content.as_bytes(), cancellation)
            .and_then(|_| temporary.sync_all());
    if let Err(error) = write_result {
        let _ = unlink_at(parent.as_raw_fd(), temporary_name.as_ptr());
        return Err(error);
    }
    drop(temporary);

    if let Err(error) = check_cancellation(cancellation) {
        let _ = unlink_at(parent.as_raw_fd(), temporary_name.as_ptr());
        return Err(error);
    }

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
fn open_directory_at(
    parent_fd: std::os::fd::RawFd,
    name: *const libc::c_char,
) -> io::Result<std::fs::File> {
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
static WRITE_TEMP_COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[cfg(not(unix))]
fn write_file_race_resistant_with_cancellation(
    workspace: &Path,
    relative_path: &str,
    content: &str,
    cancellation: Option<&ToolCancellation>,
) -> std::io::Result<()> {
    check_cancellation(cancellation)?;
    let target = workspace.join(relative_path);
    if let Some(parent) = target.parent() {
        std::fs::create_dir_all(parent)?;
    }
    check_cancellation(cancellation)?;
    std::fs::write(target, content)
}

#[cfg(unix)]
fn write_all_with_cancellation<W: Write>(
    writer: &mut W,
    content: &[u8],
    cancellation: Option<&ToolCancellation>,
) -> io::Result<()> {
    for chunk in content.chunks(16 * 1024) {
        check_cancellation(cancellation)?;
        writer.write_all(chunk)?;
    }
    check_cancellation(cancellation)
}

pub fn handle_list_dir(args: serde_json::Value, ws: &Path) -> ToolResult {
    handle_list_dir_with_cancellation(args, ws, None)
}

pub fn handle_list_dir_cancellable(
    args: serde_json::Value,
    ws: &Path,
    cancellation: &ToolCancellation,
) -> ToolResult {
    handle_list_dir_with_cancellation(args, ws, Some(cancellation))
}

fn handle_list_dir_with_cancellation(
    args: serde_json::Value,
    ws: &Path,
    cancellation: Option<&ToolCancellation>,
) -> ToolResult {
    if let Err(error) = check_cancellation(cancellation) {
        return ToolResult::err("list_dir", error.to_string());
    }
    let path_str = args
        .get("path")
        .and_then(|value| value.as_str())
        .unwrap_or(".");
    let target = match safe_join(ws, path_str) {
        Ok(target) => target,
        Err(error) => return ToolResult::err("list_dir", error),
    };

    #[cfg(unix)]
    let result = {
        let _ = target;
        list_dir_race_resistant(ws, path_str, cancellation)
    };
    #[cfg(not(unix))]
    let result = list_dir_portable(&target, cancellation);

    match result {
        Ok(content) => ToolResult::ok("list_dir", content),
        Err(error) => ToolResult::err(
            "list_dir",
            format!("Failed to list dir '{}': {}", path_str, error),
        ),
    }
}

#[cfg(unix)]
fn relative_components(relative_path: &str) -> io::Result<Vec<&std::ffi::OsStr>> {
    let mut components = Vec::new();
    for component in Path::new(relative_path).components() {
        match component {
            std::path::Component::Normal(name) => components.push(name),
            std::path::Component::CurDir => {}
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "workspace paths must contain only normal relative components",
                ))
            }
        }
    }
    Ok(components)
}

#[cfg(unix)]
fn open_relative_parent(
    workspace: &Path,
    components: &[&std::ffi::OsStr],
) -> io::Result<std::fs::File> {
    let root = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(workspace)?;
    let mut parent = root;
    for &component in components {
        let component = CString::new(component.as_bytes()).map_err(|_| {
            io::Error::new(io::ErrorKind::InvalidInput, "directory name contains NUL")
        })?;
        parent = open_directory_at(parent.as_raw_fd(), component.as_ptr())?;
    }
    Ok(parent)
}

#[cfg(unix)]
fn open_file_at(
    parent_fd: std::os::fd::RawFd,
    name: &std::ffi::OsStr,
) -> io::Result<std::fs::File> {
    let name = CString::new(name.as_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "filename contains NUL"))?;
    let fd = unsafe {
        // SAFETY: parent_fd is an open directory descriptor and name is a
        // valid NUL-terminated component.
        libc::openat(
            parent_fd,
            name.as_ptr(),
            libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        )
    };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(unsafe {
        // SAFETY: fd is newly opened and transferred to this File.
        std::fs::File::from_raw_fd(fd)
    })
}

#[cfg(unix)]
fn read_file_race_resistant(
    workspace: &Path,
    relative_path: &str,
    max_bytes: u64,
    cancellation: Option<&ToolCancellation>,
) -> io::Result<String> {
    let components = relative_components(relative_path)?;
    let file_name = components
        .last()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "empty workspace path"))?;
    let parent = open_relative_parent(workspace, &components[..components.len() - 1])?;
    let file = open_file_at(parent.as_raw_fd(), file_name)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "target is not a regular file",
        ));
    }
    if metadata.len() > max_bytes {
        return Err(io::Error::new(
            io::ErrorKind::FileTooLarge,
            format!(
                "file size ({} bytes) exceeds maximum allowed read limit ({} bytes)",
                metadata.len(),
                max_bytes
            ),
        ));
    }

    read_limited_file(file, max_bytes, cancellation)
}

#[cfg(unix)]
fn list_dir_race_resistant(
    workspace: &Path,
    relative_path: &str,
    cancellation: Option<&ToolCancellation>,
) -> io::Result<String> {
    check_cancellation(cancellation)?;
    let components = relative_components(relative_path)?;
    let directory = open_relative_parent(workspace, &components)?;
    let raw_fd = directory.into_raw_fd();
    let dir = unsafe {
        // SAFETY: raw_fd is an open directory descriptor transferred to
        // fdopendir, which takes ownership of it.
        libc::fdopendir(raw_fd)
    };
    if dir.is_null() {
        unsafe {
            // SAFETY: fdopendir failed before taking ownership of raw_fd.
            libc::close(raw_fd);
        }
        return Err(io::Error::last_os_error());
    }

    let mut names = Vec::new();
    let mut truncated = false;
    loop {
        if let Err(error) = check_cancellation(cancellation) {
            unsafe {
                // SAFETY: dir is the sole owner of the transferred directory fd.
                libc::closedir(dir);
            }
            return Err(error);
        }
        let entry = unsafe {
            // SAFETY: dir is a valid directory stream until closed below.
            libc::readdir(dir)
        };
        if entry.is_null() {
            break;
        }
        let name = unsafe {
            // SAFETY: d_name is a NUL-terminated field supplied by readdir.
            std::ffi::CStr::from_ptr((*entry).d_name.as_ptr())
        };
        let name = name.to_bytes();
        if name == b"." || name == b".." {
            continue;
        }
        if names.len() >= MAX_DIR_ENTRIES {
            truncated = true;
            break;
        }
        let clean_name: String = std::ffi::OsStr::from_bytes(name)
            .to_string_lossy()
            .chars()
            .filter(|character| !character.is_control())
            .collect();
        let mut metadata = unsafe {
            // SAFETY: zeroed memory is a valid initial value for libc::stat.
            std::mem::MaybeUninit::<libc::stat>::zeroed().assume_init()
        };
        let status = unsafe {
            // SAFETY: dir is valid, name points to the current directory
            // entry, and metadata points to writable storage.
            libc::fstatat(
                libc::dirfd(dir),
                name.as_ptr().cast(),
                &mut metadata,
                libc::AT_SYMLINK_NOFOLLOW,
            )
        };
        if status != 0 {
            continue;
        }
        let kind = if (metadata.st_mode & libc::S_IFMT) == libc::S_IFDIR {
            "dir"
        } else {
            "file"
        };
        names.push(format!("{kind}: {clean_name}"));
    }
    unsafe {
        // SAFETY: dir is the sole owner of the transferred directory fd.
        libc::closedir(dir);
    }

    names.sort();
    if truncated {
        names.push(format!(
            "... [truncated: showing first {} entries]",
            MAX_DIR_ENTRIES
        ));
    }
    Ok(names.join("\n"))
}

#[cfg(not(unix))]
fn list_dir_portable(
    target: &Path,
    cancellation: Option<&ToolCancellation>,
) -> std::io::Result<String> {
    let entries = std::fs::read_dir(target)?;
    let mut names = Vec::new();
    let mut truncated = false;
    for entry in entries.flatten() {
        check_cancellation(cancellation)?;
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
                .filter(|character| !character.is_control())
                .collect();
            names.push(format!("{kind}: {clean_name}"));
        }
    }
    names.sort();
    if truncated {
        names.push(format!(
            "... [truncated: showing first {} entries]",
            MAX_DIR_ENTRIES
        ));
    }
    Ok(names.join("\n"))
}

#[cfg(all(test, unix))]
mod tests {
    use super::{handle_list_dir, handle_read_file, write_file_race_resistant};
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

        assert!(
            result.is_ok(),
            "atomic replacement should not follow a final symlink"
        );
        assert_eq!(fs::read_to_string(&outside).unwrap(), "keep me");
        assert_eq!(
            fs::read_to_string(workspace.join("target.txt")).unwrap(),
            "overwrite me"
        );
        fs::remove_file(workspace.join("target.txt")).unwrap();
        fs::remove_file(outside).unwrap();
        fs::remove_dir(workspace).unwrap();
    }

    #[test]
    fn read_file_rejects_a_final_symlink() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock must be after the Unix epoch")
            .as_nanos();
        let workspace =
            std::env::temp_dir().join(format!("mivi-tools-read-{}-{}", std::process::id(), suffix));
        fs::create_dir(&workspace).expect("create isolated workspace");
        fs::write(workspace.join("real.txt"), "workspace secret").expect("create target file");
        symlink("real.txt", workspace.join("target.txt")).expect("create test symlink");

        let result = handle_read_file(serde_json::json!({"path": "target.txt"}), &workspace);

        assert!(!result.success);
        fs::remove_file(workspace.join("target.txt")).unwrap();
        fs::remove_file(workspace.join("real.txt")).unwrap();
        fs::remove_dir(workspace).unwrap();
    }

    #[test]
    fn list_dir_rejects_a_directory_symlink() {
        let suffix = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock must be after the Unix epoch")
            .as_nanos();
        let workspace =
            std::env::temp_dir().join(format!("mivi-tools-list-{}-{}", std::process::id(), suffix));
        fs::create_dir(&workspace).expect("create isolated workspace");
        fs::create_dir(workspace.join("real")).expect("create target directory");
        symlink("real", workspace.join("linked")).expect("create directory symlink");

        let result = handle_list_dir(serde_json::json!({"path": "linked"}), &workspace);

        assert!(!result.success);
        fs::remove_file(workspace.join("linked")).unwrap();
        fs::remove_dir(workspace.join("real")).unwrap();
        fs::remove_dir(workspace).unwrap();
    }
}
