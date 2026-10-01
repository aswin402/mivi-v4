use super::FixtureRecord;
use std::fs::{self, DirBuilder, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

pub(crate) struct ArtifactDirectory {
    path: PathBuf,
}

impl ArtifactDirectory {
    pub(crate) fn create() -> io::Result<Self> {
        let path =
            std::env::temp_dir().join(format!("mivi-fixture-capture-{}", uuid::Uuid::new_v4()));
        let mut builder = DirBuilder::new();
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        builder.create(&path)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&path, fs::Permissions::from_mode(0o700))?;
        }
        Ok(Self { path })
    }

    pub(crate) fn write(&self, sequence: usize, record: &FixtureRecord) -> io::Result<()> {
        self.write_with(sequence, record, |file, record| {
            serde_json::to_writer(file, record).map_err(io::Error::other)
        })
    }

    fn write_with(
        &self,
        sequence: usize,
        record: &FixtureRecord,
        write_record: impl FnOnce(&mut fs::File, &FixtureRecord) -> io::Result<()>,
    ) -> io::Result<()> {
        let path = self.path.join(format!("request-{sequence:04}.json"));
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        let mut file = options.open(path)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            file.set_permissions(fs::Permissions::from_mode(0o600))?;
        }
        write_record(&mut file, record)?;
        file.flush()
    }

    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fixture_diagnostics::{FixtureLimits, FixtureSession};

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "mivi-fixture-artifact-test-{}",
                uuid::Uuid::new_v4()
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn record() -> FixtureRecord {
        let session = FixtureSession::new(FixtureLimits::default()).unwrap();
        let sequence = session.arm("artifact").unwrap();
        session
            .begin(
                sequence,
                "synthetic",
                "",
                serde_json::json!({"max_tokens": 1}),
                serde_json::json!({"model_config_name": {"value": "synthetic", "clipped": false}}),
            )
            .unwrap()
    }

    fn directory_at(path: PathBuf) -> ArtifactDirectory {
        ArtifactDirectory { path }
    }

    #[test]
    fn creates_private_directory_and_exclusive_private_files() {
        let directory = ArtifactDirectory::create().unwrap();
        let record = record();
        directory.write(1, &record).unwrap();
        assert_eq!(
            directory.write(1, &record).unwrap_err().kind(),
            io::ErrorKind::AlreadyExists
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(directory.path()).unwrap().permissions().mode() & 0o777,
                0o700
            );
            assert_eq!(
                fs::metadata(directory.path().join("request-0001.json"))
                    .unwrap()
                    .permissions()
                    .mode()
                    & 0o777,
                0o600
            );
        }
        fs::remove_dir_all(directory.path()).unwrap();
    }

    #[cfg(unix)]
    #[test]
    fn refuses_preexisting_symlink_filename() {
        use std::os::unix::fs::symlink;
        let temp = TestDirectory::new();
        let target = temp.0.join("target");
        fs::write(&target, b"keep").unwrap();
        let directory = directory_at(temp.0.clone());
        symlink(&target, directory.path().join("request-0001.json")).unwrap();
        assert_eq!(
            directory.write(1, &record()).unwrap_err().kind(),
            io::ErrorKind::AlreadyExists
        );
        assert_eq!(fs::read(target).unwrap(), b"keep");
    }

    #[test]
    fn injected_write_failure_is_separate_from_record_outcome() {
        let temp = TestDirectory::new();
        let directory = directory_at(temp.0.clone());
        let mut record = record();
        record.engine_terminal = crate::fixture_diagnostics::EngineTerminal::ModelError;
        let outcome = record.engine_terminal;
        let error = directory.write_with(1, &record, |_file, _record| {
            Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "synthetic write failure",
            ))
        });
        assert_eq!(error.unwrap_err().kind(), io::ErrorKind::WriteZero);
        assert_eq!(record.engine_terminal, outcome);
        assert!(directory.path().join("request-0001.json").exists());
    }
}
