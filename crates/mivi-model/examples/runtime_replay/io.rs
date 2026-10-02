use mivi_model::fixture_diagnostics::replay::ReplayInput;
use std::fs::{File, OpenOptions};
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
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK);
    }
    let file = options.open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file() || metadata.len() > MAX_FILE_BYTES as u64 {
        return Err(io::Error::other("invalid replay input file"));
    }
    let mut bytes = Vec::new();
    file.take(MAX_FILE_BYTES as u64 + 1)
        .read_to_end(&mut bytes)?;
    if bytes.len() > MAX_FILE_BYTES {
        return Err(io::Error::other("replay input exceeds byte limit"));
    }
    let input: ReplayInput =
        serde_json::from_slice(&bytes).map_err(|_| io::Error::other("invalid replay JSON"))?;
    input.validate().map_err(io::Error::other)?;
    Ok(input)
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
