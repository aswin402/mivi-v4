//! Private diagnostic executable, never part of the normal server CLI.
#[path = "runtime_replay/generation.rs"]
mod generation;
#[path = "runtime_replay/io.rs"]
mod private_io;

use mivi_model::fixture_diagnostics::replay::ReplayInput;
use std::error::Error;
use std::ffi::OsString;
use std::path::Path;
use std::path::PathBuf;
use std::time::Instant;

struct Args {
    model: PathBuf,
    input: PathBuf,
    output: PathBuf,
}

fn parse_args(args: impl IntoIterator<Item = OsString>) -> Result<Args, &'static str> {
    let mut args = args.into_iter();
    let (mut model, mut input, mut output) = (None, None, None);
    while let Some(flag) = args.next() {
        let slot = match flag.to_str() {
            Some("--model") => &mut model,
            Some("--input") => &mut input,
            Some("--output") => &mut output,
            _ => return Err("unknown argument"),
        };
        if slot.is_some() {
            return Err("duplicate argument");
        }
        let value = args.next().ok_or("missing argument value")?;
        if value.is_empty() || value.to_str().is_some_and(|s| s.starts_with("--")) {
            return Err("invalid argument value");
        }
        *slot = Some(PathBuf::from(value));
    }
    Ok(Args {
        model: model.ok_or("missing model")?,
        input: input.ok_or("missing input")?,
        output: output.ok_or("missing output")?,
    })
}

#[cfg(not(test))]
fn main() -> std::process::ExitCode {
    let started = Instant::now();
    let args = match parse_args(std::env::args_os().skip(1)) {
        Ok(args) => args,
        Err(reason) => {
            eprintln!("runtime replay: {reason}; usage: --model PATH --input PATH --output PATH");
            return std::process::ExitCode::FAILURE;
        }
    };
    if execute_with(args, started, generation::run).is_err() {
        eprintln!("runtime replay failed; check input bounds, private output location, and any private result record");
        std::process::ExitCode::FAILURE
    } else {
        println!("runtime replay complete; private result written");
        std::process::ExitCode::SUCCESS
    }
}

fn execute_with(
    args: Args,
    started: Instant,
    run: impl FnOnce(&Path, ReplayInput, Instant) -> Result<serde_json::Value, Box<dyn Error>> + Send,
) -> Result<(), Box<dyn Error>> {
    let input = private_io::read_input(&args.input)?;
    let mut output = private_io::PrivateOutput::create(&args.output)?;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(2).build()?;
    let result =
        pool.install(move || run(&args.model, input, started).map_err(|error| error.to_string()));
    let mut value = match result {
        Ok(value) if value.is_object() => value,
        Ok(_) => {
            serde_json::json!({"schema":1,"status":"model_error","error":"invalid replay record"})
        }
        Err(error) => serde_json::json!({"schema":1,"status":"model_error","error":error}),
    };
    value["model_call_returned"] = serde_json::json!(true);
    value["process_exit"] = serde_json::json!("not_observed");
    value["runner_elapsed_us"] = serde_json::json!(u64::try_from(started.elapsed().as_micros())?);
    output.write_json(&value)?;
    if value["status"] == "complete" {
        Ok(())
    } else {
        Err("replay did not complete".into())
    }
}

#[cfg(test)]
mod cli_io_tests {
    use super::*;
    use std::fs;
    use std::io::Write;
    use tempfile::TempDir;

    fn argv(values: &[&str]) -> Vec<OsString> {
        values.iter().map(OsString::from).collect()
    }

    fn input_bytes() -> &'static [u8] {
        br#"{"prompt_ids":[1,2],"context":128,"tile":64,"max_tokens":16,"profile":false,"split_prefill":false,"teacher_forced_ids":[],"logit_ids":[]}"#
    }

    #[test]
    fn parses_required_paths_and_rejects_invalid_arguments() {
        let args = parse_args(argv(&[
            "--model", "a.gguf", "--input", "in.json", "--output", "out.json",
        ]))
        .unwrap();
        assert_eq!(args.model, PathBuf::from("a.gguf"));
        assert_eq!(args.input, PathBuf::from("in.json"));
        assert_eq!(args.output, PathBuf::from("out.json"));
        for values in [
            vec![],
            vec!["--model"],
            vec!["--model", "a", "--input", "b"],
            vec!["--unknown", "x"],
            vec![
                "--model", "a", "--model", "b", "--input", "c", "--output", "d",
            ],
        ] {
            assert!(parse_args(argv(&values)).is_err());
        }
    }

    #[test]
    fn reads_bounded_valid_input_and_rejects_oversized_or_invalid_json() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("input.json");
        fs::write(&path, input_bytes()).unwrap();
        let parsed: ReplayInput = private_io::read_input(&path).unwrap();
        assert_eq!(parsed.prompt_ids, [1, 2]);
        fs::write(&path, b"not JSON").unwrap();
        assert!(private_io::read_input(&path).is_err());
        fs::File::create(&path)
            .unwrap()
            .set_len(private_io::MAX_FILE_BYTES as u64 + 1)
            .unwrap();
        assert!(private_io::read_input(&path).is_err());
        assert!(private_io::read_input(dir.path()).is_err());
    }

    #[cfg(unix)]
    fn private_dir() -> TempDir {
        use std::os::unix::fs::PermissionsExt;
        let dir = TempDir::new().unwrap();
        fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o700)).unwrap();
        dir
    }

    #[cfg(unix)]
    #[test]
    fn creates_private_output_and_refuses_existing_files() {
        use std::os::unix::fs::PermissionsExt;
        let dir = private_dir();
        let path = dir.path().join("result.json");
        let mut output = private_io::PrivateOutput::create(&path).unwrap();
        output.write_json(&serde_json::json!({"schema":1})).unwrap();
        assert_eq!(
            fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        let before = fs::read(&path).unwrap();
        assert!(private_io::PrivateOutput::create(&path).is_err());
        assert_eq!(fs::read(&path).unwrap(), before);
    }

    #[cfg(unix)]
    #[test]
    fn refuses_nonprivate_parent_and_symlinks_without_overwriting_target() {
        use std::os::unix::fs::{symlink, PermissionsExt};
        let dir = private_dir();
        let target = dir.path().join("target");
        fs::write(&target, input_bytes()).unwrap();
        let link = dir.path().join("link");
        symlink(&target, &link).unwrap();
        assert!(private_io::read_input(&link).is_err());
        assert!(private_io::PrivateOutput::create(&link).is_err());
        assert_eq!(fs::read(&target).unwrap(), input_bytes());
        let parent_link = dir.path().join("parent-link");
        symlink(dir.path(), &parent_link).unwrap();
        assert!(private_io::PrivateOutput::create(&parent_link.join("escape.json")).is_err());
        fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o755)).unwrap();
        assert!(private_io::PrivateOutput::create(&dir.path().join("public.json")).is_err());
    }

    #[test]
    fn serialization_budget_rejects_oversize_before_file_write() {
        let mut bytes = private_io::BoundedBytes::new(3);
        bytes.write_all(b"abc").unwrap();
        assert!(bytes.write_all(b"d").is_err());
        assert_eq!(bytes.as_slice(), b"abc");
    }

    #[cfg(unix)]
    #[test]
    fn opens_directory_chain_without_following_any_symlink() {
        use std::os::unix::fs::symlink;
        let dir = private_dir();
        let nested = dir.path().join("nested");
        fs::create_dir(&nested).unwrap();
        assert!(private_io::open_dir_no_symlinks(&nested).is_ok());
        let link = dir.path().join("link");
        symlink(&nested, &link).unwrap();
        assert!(private_io::open_dir_no_symlinks(&link).is_err());
        assert!(private_io::open_dir_no_symlinks(&link.join("nested")).is_err());
    }

    #[cfg(unix)]
    #[test]
    fn oversized_serialization_leaves_output_empty() {
        let dir = private_dir();
        let path = dir.path().join("result.json");
        let mut output = private_io::PrivateOutput::create(&path).unwrap();
        let value = serde_json::json!({"text":"a".repeat(private_io::MAX_FILE_BYTES)});
        assert!(output.write_json(&value).is_err());
        assert_eq!(fs::metadata(path).unwrap().len(), 0);
    }

    #[cfg(unix)]
    #[test]
    fn result_writer_rejects_second_record_without_appending() {
        let dir = private_dir();
        let path = dir.path().join("result.json");
        let mut output = private_io::PrivateOutput::create(&path).unwrap();
        output.write_json(&serde_json::json!({"schema":1})).unwrap();
        let before = fs::read(&path).unwrap();
        assert!(output.write_json(&serde_json::json!({"schema":2})).is_err());
        assert_eq!(fs::read(path).unwrap(), before);
    }

    #[cfg(unix)]
    #[test]
    fn preflight_rejects_input_or_output_before_loading_model() {
        let dir = private_dir();
        let input = dir.path().join("input.json");
        let output = dir.path().join("result.json");
        fs::write(&input, b"invalid JSON").unwrap();
        let args = Args {
            model: PathBuf::from("does-not-exist"),
            input: input.clone(),
            output: output.clone(),
        };
        assert!(execute_with(args, Instant::now(), |_, _, _| panic!(
            "model must not load"
        ))
        .is_err());
        assert!(!output.exists());
        fs::write(&input, input_bytes()).unwrap();
        fs::write(&output, b"preserve me").unwrap();
        let args = Args {
            model: PathBuf::from("does-not-exist"),
            input,
            output: output.clone(),
        };
        assert!(execute_with(args, Instant::now(), |_, _, _| panic!(
            "model must not load"
        ))
        .is_err());
        assert_eq!(fs::read(output).unwrap(), b"preserve me");
    }

    #[cfg(unix)]
    #[test]
    fn executor_uses_two_threads_and_records_return_without_claiming_process_exit() {
        let dir = private_dir();
        let input = dir.path().join("input.json");
        let output = dir.path().join("result.json");
        fs::write(&input, input_bytes()).unwrap();
        let args = Args {
            model: PathBuf::from("synthetic"),
            input,
            output: output.clone(),
        };
        execute_with(args, Instant::now(), |_, _, _| {
            assert_eq!(rayon::current_num_threads(), 2);
            Ok(serde_json::json!({"schema":1,"status":"complete"}))
        })
        .unwrap();
        let result: serde_json::Value = serde_json::from_slice(&fs::read(output).unwrap()).unwrap();
        assert_eq!(result["model_call_returned"], true);
        assert_eq!(result["process_exit"], "not_observed");
    }

    #[cfg(unix)]
    #[test]
    fn failed_or_cancelled_run_writes_private_record_and_returns_failure() {
        let dir = private_dir();
        let input = dir.path().join("input.json");
        fs::write(&input, input_bytes()).unwrap();
        for (index, status) in ["cancelled", "partial"].iter().enumerate() {
            let output = dir.path().join(format!("result-{index}.json"));
            let args = Args {
                model: PathBuf::from("synthetic"),
                input: input.clone(),
                output: output.clone(),
            };
            assert!(execute_with(args, Instant::now(), |_, _, _| Ok(
                serde_json::json!({"schema":1,"status":status})
            ))
            .is_err());
            let result: serde_json::Value =
                serde_json::from_slice(&fs::read(output).unwrap()).unwrap();
            assert_eq!(result["status"], *status);
        }
        let output = dir.path().join("error.json");
        let args = Args {
            model: PathBuf::from("synthetic"),
            input,
            output: output.clone(),
        };
        assert!(execute_with(args, Instant::now(), |_, _, _| Err(
            "synthetic failure".into()
        ))
        .is_err());
        let result: serde_json::Value = serde_json::from_slice(&fs::read(output).unwrap()).unwrap();
        assert_eq!(result["status"], "model_error");
    }
}
