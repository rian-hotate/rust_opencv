extern crate glob;

use std::collections::VecDeque;
use std::path::PathBuf;
use std::env::*;
use glob::glob;
use std::sync::mpsc;

use sample_opencv::image_detector::*;
use sample_opencv::movie_detector::*;

enum Operation {
    image,
    movie
}

fn input_filenames(operation: Operation, ext: &str) -> Result<VecDeque<PathBuf>, std::io::Error> {
    let current_path = current_dir().unwrap();
    let mut input_path: String = String::from(current_path.to_str().unwrap());

    let mut v = VecDeque::new();
    match operation {
        Operation::image => {
            let _ = input_path.push_str("/input_images/*");
            let _ = input_path.push_str(ext);

            for path in glob(&input_path).unwrap().filter_map(Result::ok) {
                let _ = v.push_front(path);
            }
        },
        Operation::movie => {
            let _ = input_path.push_str("/input_movies/*");
            let _ = input_path.push_str(ext);

            for path in glob(&input_path).unwrap().filter_map(Result::ok) {
                let _ = v.push_front(path);
            }
        },
        _ => {},
    }
    Ok(v)
}


fn main() {
    let args: Vec<String> = args().collect();
    if args.len() == 2 {
        if args[1] == "-h" {
            println!("Usage: [OPTION]");
            println!("  image:  Face detection is performed for images under /input_images");
            println!("  movie:  Face detection is performed for movies under /input_movies");
        } else if args[1] == "image" {
            let (tx, rx): (std::sync::mpsc::Sender<bool>, std::sync::mpsc::Receiver<bool>) = mpsc::channel();
            let tx_clone = tx.clone();

            let mut operation = ImageFaceDetecter::new(tx);
            let _ = operation.run(input_filenames(Operation::image, ".jpg").unwrap());

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
        } else if args[1] == "movie" {
            let (tx, rx): (std::sync::mpsc::Sender<bool>, std::sync::mpsc::Receiver<bool>) = mpsc::channel();
            let tx_clone = tx.clone();

            let mut operation = MovieFaceDetecter::new(tx);
            let _ = operation.run(input_filenames(Operation::movie, ".mp4").unwrap());

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
    } else {
        println!("missing operand");
        println!("Try '-h' for more information.");
    }
}
