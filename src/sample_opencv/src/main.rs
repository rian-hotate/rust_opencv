extern crate glob;

use std::collections::VecDeque;
use std::path::PathBuf;
use std::env::*;
use glob::glob;
use std::sync::mpsc;

use sample_opencv::detector::*;

fn input_filenames(ext: &str) -> Result<VecDeque<PathBuf>, std::io::Error> {
    let current_path = current_dir().unwrap();
    let mut input_path: String = String::from(current_path.to_str().unwrap());

    let _ = input_path.push_str("/input_images/*");
    let _ = input_path.push_str(ext);
    
    let mut v = VecDeque::new();
    for path in glob(&input_path).unwrap().filter_map(Result::ok) {
        let _ = v.push_front(path);
    }
    Ok(v)
}


fn main() {
    let args: Vec<String> = args().collect();
    if (args.len() > 1) {
        let (tx, rx): (std::sync::mpsc::Sender<bool>, std::sync::mpsc::Receiver<bool>) = mpsc::channel();
        let tx_clone = tx.clone();

        let mut operation = FaceDetecter::new(tx);
        let _ = operation.run(input_filenames(".jpg").unwrap());

        ctrlc::set_handler(move || {
            let _ = operation.stop();
            let _ = operation.join();
            let _ = tx_clone.send(true);
        }).expect("Error setting Ctrl-C handler");

        loop {
            match rx.try_recv() {
                Ok(true) => {
                    println!("Operation Finish!");
                    break;
                }
                _ => {}
            }
        }
    } else {
        println!("missing operand");
        println!("Try '-h' for more information.");
    }
}
