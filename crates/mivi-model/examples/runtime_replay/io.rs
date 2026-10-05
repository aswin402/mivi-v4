use mivi_model::fixture_diagnostics::replay::ReplayInput;
use serde::de::DeserializeOwned;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::Path;
pub const MAX_FILE_BYTES: usize = 4 * 1024 * 1024;
#[cfg(unix)]
pub(super) fn open_dir_no_symlinks(path: &Path) -> io::Result<File> {
    use std::ffi::CString;
    use std::os::fd::{AsRawFd, FromRawFd};
    use std::os::unix::ffi::OsStrExt;
    use std::path::Component;
    if !path.is_absolute() {
        return Err(io::Error::other("directory walk requires an absolute path"));
    }
    let mut directory = File::open("/")?;
    for component in path.components() {
        let name = match component {
            Component::RootDir => continue,
            Component::Normal(name) => name,
            _ => return Err(io::Error::other("directory traversal is not allowed")),
        };
        let name = CString::new(name.as_bytes())
            .map_err(|_| io::Error::other("invalid directory name"))?;
        // SAFETY: both descriptor and C string remain live during openat;
        // each basename is opened relative to a pinned directory, never via
        // a path lookup following an intermediate symlink.
        let fd = unsafe {
            libc::openat(
                directory.as_raw_fd(),
                name.as_ptr(),
                libc::O_RDONLY | libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC,
            )
        };
        if fd < 0 {
            return Err(io::Error::last_os_error());
        }
        // SAFETY: a fresh directory fd is transferred into exactly one owner.
        directory = unsafe { File::from_raw_fd(fd) };
    }
    Ok(directory)
}
pub fn read_input(path: &Path) -> io::Result<ReplayInput> {
    let input: ReplayInput = read_json(path)?;
    input.validate().map_err(io::Error::other)?;
    Ok(input)
}

pub fn read_json<T: DeserializeOwned>(path: &Path) -> io::Result<T> {
    let file = open_bounded_regular_file(path, MAX_FILE_BYTES as u64)?;
    let mut bytes = Vec::new();
    file.take(MAX_FILE_BYTES as u64 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > MAX_FILE_BYTES {
        return Err(io::Error::other("JSON input exceeds byte limit"));
    }
    serde_json::from_slice(&bytes).map_err(|_| io::Error::other("invalid JSON input"))
}

#[cfg(unix)]
pub fn open_bounded_regular_file(path: &Path, max_bytes: u64) -> io::Result<File> {
    use std::ffi::CString;
    use std::os::fd::{AsRawFd, FromRawFd};
    use std::os::unix::ffi::OsStrExt;
    use std::path::Component;

    if path.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err(io::Error::other("file input traversal is not allowed"));
    }
    let absolute_path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let parent = absolute_path
        .parent()
        .ok_or_else(|| io::Error::other("missing file input parent"))?;
    let directory = open_dir_no_symlinks(parent)?;
    let name = absolute_path
        .file_name()
        .ok_or_else(|| io::Error::other("missing file input filename"))?;
    let name = CString::new(name.as_bytes())
        .map_err(|_| io::Error::other("invalid file input filename"))?;
    // SAFETY: both the pinned directory descriptor and basename remain live
    // for openat; O_NOFOLLOW refuses a final-component symlink.
    let fd = unsafe {
        libc::openat(
            directory.as_raw_fd(),
            name.as_ptr(),
            libc::O_RDONLY | libc::O_NOFOLLOW | libc::O_NONBLOCK | libc::O_CLOEXEC,
        )
    };
    if fd < 0 {
        return Err(io::Error::last_os_error());
    }
    // SAFETY: fd is a newly opened descriptor transferred to one File owner.
    let file = unsafe { File::from_raw_fd(fd) };
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() > max_bytes {
        return Err(io::Error::other(
            "input must be a regular file within its byte limit",
        ));
    }
    Ok(file)
}

#[cfg(not(unix))]
pub fn open_bounded_regular_file(_path: &Path, _max_bytes: u64) -> io::Result<File> {
    Err(io::Error::other(
        "bounded private input requires Unix descriptor pinning",
    ))
}
pub struct PrivateOutput {
    file: File,
    attempted: bool,
}
impl PrivateOutput {
    pub fn create(path: &Path) -> io::Result<Self> {
        #[cfg(unix)]
        {
            use std::ffi::CString;
            use std::os::fd::{AsRawFd, FromRawFd};
            use std::os::unix::ffi::OsStrExt;
            use std::os::unix::fs::{MetadataExt, PermissionsExt};
            use std::path::Component;
            let path = if path.is_absolute() {
                path.to_path_buf()
            } else {
                std::env::current_dir()?.join(path)
            };
            if path.components().any(|c| matches!(c, Component::ParentDir)) {
                return Err(io::Error::other("parent traversal is not allowed"));
            }
            let parent = path
                .parent()
                .ok_or_else(|| io::Error::other("missing output parent"))?;
            let directory = open_dir_no_symlinks(parent)?;
            let metadata = directory.metadata()?;
            // SAFETY: geteuid has no arguments or memory-access preconditions.
            let uid = unsafe { libc::geteuid() };
            if metadata.permissions().mode() & 0o777 != 0o700 || metadata.uid() != uid {
                return Err(io::Error::other(
                    "output parent must be owned and private (0700)",
                ));
            }
            let name = path
                .file_name()
                .ok_or_else(|| io::Error::other("missing output filename"))?;
            let name = CString::new(name.as_bytes())
                .map_err(|_| io::Error::other("invalid output filename"))?;
            // SAFETY: the directory descriptor and NUL-terminated basename are
            // live; O_EXCL refuses existing files/symlinks. openat pins the
            // verified private directory even if its path is later renamed.
            let fd = unsafe {
                libc::openat(
                    directory.as_raw_fd(),
                    name.as_ptr(),
                    libc::O_WRONLY
                        | libc::O_CREAT
                        | libc::O_EXCL
                        | libc::O_NOFOLLOW
                        | libc::O_CLOEXEC,
                    0o600 as libc::mode_t,
                )
            };
            if fd < 0 {
                return Err(io::Error::last_os_error());
            }
            // SAFETY: fd is a fresh owned descriptor, transferred exactly once.
            let file = unsafe { File::from_raw_fd(fd) };
            file.set_permissions(std::fs::Permissions::from_mode(0o600))?;
            Ok(Self {
                file,
                attempted: false,
            })
        }
        #[cfg(not(unix))]
        {
            let _ = path;
            Err(io::Error::other(
                "private replay output requires Unix permission enforcement",
            ))
        }
    }
    pub fn write_json(&mut self, value: &serde_json::Value) -> io::Result<()> {
        if self.attempted {
            return Err(io::Error::other("replay output is single-use"));
        }
        self.attempted = true;
        let mut bytes = BoundedBytes::new(MAX_FILE_BYTES);
        serde_json::to_writer(&mut bytes, value).map_err(io::Error::other)?;
        self.file.write_all(bytes.as_slice())?;
        self.file.flush()
    }
}
pub struct BoundedBytes {
    bytes: Vec<u8>,
    cap: usize,
}
impl BoundedBytes {
    pub fn new(cap: usize) -> Self {
        Self {
            bytes: Vec::new(),
            cap,
        }
    }
    pub fn as_slice(&self) -> &[u8] {
        &self.bytes
    }
}
impl Write for BoundedBytes {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > self.cap.saturating_sub(self.bytes.len()) {
            return Err(io::Error::other("serialized replay exceeds byte limit"));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod generic_reader_tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Deserialize, PartialEq)]
    struct GenericRecord {
        count: usize,
    }

    #[test]
    fn generic_json_reader_deserializes_bounded_regular_files() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("record.json");
        std::fs::write(&path, br#"{"count":7}"#).unwrap();
        assert_eq!(
            read_json::<GenericRecord>(&path).unwrap(),
            GenericRecord { count: 7 }
        );
        std::fs::File::create(&path)
            .unwrap()
            .set_len(MAX_FILE_BYTES as u64 + 1)
            .unwrap();
        assert!(read_json::<GenericRecord>(&path).is_err());
        assert!(read_json::<GenericRecord>(dir.path()).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn generic_json_reader_refuses_parent_and_file_symlinks() {
        use std::os::unix::fs::symlink;
        let dir = tempfile::tempdir().unwrap();
        let real_parent = dir.path().join("real");
        std::fs::create_dir(&real_parent).unwrap();
        let target = real_parent.join("record.json");
        std::fs::write(&target, br#"{"count":7}"#).unwrap();
        let file_link = real_parent.join("file-link.json");
        symlink(&target, &file_link).unwrap();
        assert!(read_json::<GenericRecord>(&file_link).is_err());
        let parent_link = dir.path().join("parent-link");
        symlink(&real_parent, &parent_link).unwrap();
        assert!(read_json::<GenericRecord>(&parent_link.join("record.json")).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn bounded_regular_file_opener_pins_regular_inode_and_enforces_size_cap() {
        use std::os::unix::fs::symlink;
        let dir = tempfile::tempdir().unwrap();
        let regular = dir.path().join("regular.json");
        std::fs::write(&regular, br#"{"count":7}"#).unwrap();
        let opened = open_bounded_regular_file(&regular, 32).unwrap();
        assert_eq!(opened.metadata().unwrap().len(), 11);

        let link = dir.path().join("link.json");
        symlink(&regular, &link).unwrap();
        assert!(open_bounded_regular_file(&link, 32).is_err());
        assert!(open_bounded_regular_file(&regular, 10).is_err());
    }
}
