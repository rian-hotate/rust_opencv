#![feature(conservative_impl_trait, universal_impl_trait)]
extern crate glob;

use std::env::*;
use std::path::PathBuf;
use opencv::imgcodecs::*;
use opencv::core::*;
use opencv::objdetect::*;
use opencv::*;
use glob::glob;


fn main() {
    let ext = ".jpg";
    let paths = input_filenames(ext).unwrap();
    let imgs = load_img(paths).unwrap();
    output_imgs(faceDetection(imgs, ext).unwrap(), ext);
}

fn load_img(paths: Vec<PathBuf>) -> Result<Vec<Mat>, ()> {
    let mut v = Vec::new();
    for path in paths {
        v.push(imread(&path.into_os_string().into_string().unwrap(), IMREAD_UNCHANGED).unwrap());
    }
    Ok(v)
}

pub fn input_filenames(ext: &str) -> Result<Vec<PathBuf>, std::io::Error> {
    let current_path = current_dir().unwrap();
    let mut input_path: String = String::from(current_path.to_str().unwrap());

    input_path.push_str("/input_images/*");
    input_path.push_str(ext);
    
    let mut v = Vec::new();
    for path in glob(&input_path).unwrap().filter_map(Result::ok) {
        v.push(path);
    }
    Ok(v)
}

pub fn output_filenames(ext: &str) -> Result<Vec<String>, std::io::Error> {
    let paths = input_filenames(ext).unwrap();

    let mut v = Vec::new();
    for path in paths {
        v.push(path.to_str().unwrap().replace("/input_images/", "/output_images/").replace(ext, &format!("{}{}", "_out", ext)));
    }
    Ok(v)
}


fn output_imgs(imgs: Vec<Mat>, ext: &str) {
    let mut v = Vector::new();
    v.insert(0, IMWRITE_JPEG_CHROMA_QUALITY);

    let paths = output_filenames(ext).unwrap();
    for (index, img) in imgs.iter().enumerate() {
        imwrite(&paths[index], &img, &v).unwrap();
    }
}

fn faceDetection(imgs: Vec<Mat>, ext: &str) -> Result<Vec<Mat>, opencv::Error> {

    let mut face_cascade: CascadeClassifier = CascadeClassifier::new("/work/opencv/data/haarcascades/haarcascade_frontalface_default.xml").unwrap();

    let mut v = Vec::new();
    for img in imgs.iter() {
        let mut src_img = Mat::copy(&img)?;
        let mut gray = Mat::default()?;
        imgproc::cvt_color(img, &mut gray, imgproc::COLOR_BGR2GRAY, 0);
        let mut faces = types::VectorOfRect::new();

        face_cascade.detect_multi_scale(&gray, &mut faces, 1.3, 5, 0, Size_::new(150, 150), Size_::new(150, 150));

        for rect in faces.iter() {
            imgproc::rectangle(&mut src_img, rect, Scalar_::new(255.0, 0.0, 0.0, 3.0), 1, 8, 0);
        }
        v.push(src_img);
    }
    Ok(v)
}
