#include "jniheaders.h"
#include "mtcnncrop.h"

/**
 * JNI wrapper to load the various mtcnn detector model and image
 * 
 * Convert image from base64, pass the caffepath using Struct caffe path
 * Return cropped base64 of the face in input image 
 * */
JNIEXPORT jstring JNICALL Java_com_sample_facedetect_MtcnnDetector_cropFace
  (JNIEnv * env, jobject obj, 
  jstring imagebase64, 
  jstring pnetcaffe, 
  jstring pnetprototxt, 
  jstring rnetcaffe, 
  jstring rnetprototxt, 
  jstring onetcaffe, 
  jstring onetprototxt){


    const char *pnetcaffepath;
    const char *pnetprotopath;
    pnetcaffepath = env->GetStringUTFChars(pnetcaffe, 0);
    pnetprotopath = env->GetStringUTFChars(pnetprototxt, 0);


    const char *rnetcaffepath;
    const char *rnetprotopath;
    rnetcaffepath = env->GetStringUTFChars(rnetcaffe, 0);
    rnetprotopath = env->GetStringUTFChars(rnetprototxt, 0);

    const char *onetcaffepath;
    const char *onetprotopath;
    onetcaffepath = env->GetStringUTFChars(onetcaffe, 0);
    onetprotopath = env->GetStringUTFChars(onetprototxt, 0);

    if(pnetcaffepath == NULL || pnetprotopath == NULL || 
      rnetcaffepath == NULL || rnetprotopath == NULL || 
      onetcaffepath == NULL || onetprotopath == NULL) {
      throw  std::invalid_argument("Model fileapath cannot be none");
    }


    std::string pnetcaffefilename(pnetcaffepath);
    std::string pnetprotofilename(pnetprotopath);

    std::string rnetcaffefilename(rnetcaffepath);
    std::string rnetprotofilename(rnetprotopath);

    std::string onetcaffefilename(onetcaffepath);
    std::string onetprotofilename(onetprotopath);


    ModelPaths det1 = {pnetcaffefilename, pnetprotofilename};
    ModelPaths det2 = {rnetcaffefilename, rnetprotofilename};
    ModelPaths det3 = {onetcaffefilename, onetprotofilename};


    MtcnnCrop mtcnn(det1, det2, det3);


    const char *imagestring_char;
    imagestring_char = env->GetStringUTFChars(imagebase64, 0);

    if(imagestring_char == NULL) {
      throw std::invalid_argument("Invalid string passed for image string");
    }

    cv::Mat image = mtcnn.readBase64Image(imagestring_char);


    cv::Mat croppedFace = mtcnn.cropface(image);


    //Convert cropped face to base64
    std::string decodedface = mtcnn.imagetobase64(croppedFace);

    char decodedfacearray[decodedface.length() + 1];  // Make sure there's enough space
    strcpy(decodedfacearray, decodedface.c_str());

    return env->NewStringUTF(decodedfacearray);
}