use std::env;
use std::path::Path;
use std::process::ExitCode;

fn main() -> ExitCode {
    let mut args = env::args().skip(1);
    let Some(path) = args.next() else {
        eprintln!("usage: sentinel-parse FILE");
        return ExitCode::from(2);
    };
    if path == "-h" || path == "--help" {
        println!("usage: sentinel-parse FILE");
        println!("Prints a JSON description of a model file. Weights are not loaded and pickle is not executed.");
        return ExitCode::SUCCESS;
    }
    let report = sentinel_native::scan_path(Path::new(&path));
    match serde_json::to_string(&report) {
        Ok(json) => println!("{json}"),
        Err(e) => {
            eprintln!("json: {e}");
            return ExitCode::from(2);
        }
    }
    if report.error.is_some() {
        ExitCode::from(2)
    } else {
        ExitCode::SUCCESS
    }
}
