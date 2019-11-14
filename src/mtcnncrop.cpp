#include <iostream>
#include "mtcnncrop.h"


MtcnnCrop::MtcnnCrop() {

    std::cout << "---Mtcnncrop initialized" << std::endl;
}

MtcnnCrop::~MtcnnCrop() {

    std::cout << "--Mtcnncrop destroyed" << std::endl;

}

const std::string MtcnnCrop::base64_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";


/**
 * Construct the detector instance once file path are sent to the jni
 * */
MtcnnCrop::MtcnnCrop(ModelPaths det1, ModelPaths det2, ModelPaths det3) {

    ProposalNetwork::Config pConfig;
    pConfig.caffeModel = det1.detcaffepath;
    pConfig.protoText = det1.detprototxtpath;
    pConfig.threshold = 0.6f;


    RefineNetwork::Config rConfig;
    rConfig.caffeModel = det2.detcaffepath;
    rConfig.protoText = det2.detprototxtpath;
    rConfig.threshold = 0.7f;

    OutputNetwork::Config oConfig;
    oConfig.caffeModel = det3.detcaffepath;
    oConfig.protoText = det3.detprototxtpath;
    oConfig.threshold = 0.7f;


    _mtcnndetector = std::unique_ptr<MTCNNDetector>(new MTCNNDetector(pConfig, rConfig, oConfig));

}

/**
 * Using mtcnn detector loaded from the configuratio file, crop a face that is expected to be only one face
 * */
cv::Mat MtcnnCrop::cropface(cv::Mat image){

    std::vector<Face> faces;

    faces = this->_mtcnndetector->detect(image, 20.f, 0.709f);

    if(faces.size() < 1) {
        throw std::length_error("No face found in image");
    }
    if(faces.size() > 1) {
        throw std::length_error("Multiple face found");
    }

    auto rect = faces[0].bbox.getRect();

    return this->cropwithrect(image, rect);
}



cv::Mat MtcnnCrop::cropwithrect(const cv::Mat image, cv::Rect rect){
    return image(rect).clone();
}


/**
 * Check if a character of a base 64 is base 64, checking if it is alphanumeric
 * or + or /
 * */
bool MtcnnCrop::is_base64(unsigned char c) {
    return (isalnum(c) || (c == '+') || (c == '/'));
}

/**
 * Decode a base64 Image string
 * 
 * */
 std::string MtcnnCrop::base64_decode(std::string const & encoded_string){

     int in_len = encoded_string.size();
     int i = 0;
     int j = 0;
     int in_ = 0;

     unsigned char char_array_4[4], char_array_3[3];
     std::string ret;


     while (in_len-- && (encoded_string[in_] != '=') && is_base64(encoded_string[in_])) {

         char_array_4[i++] = encoded_string[in_]; in_++;
         if (i == 4) {
             for (i=0; i <4; i++) {
                 char_array_4[i] = this->base64_chars.find(char_array_4[i]); //String find
             }
             char_array_3[0] = ( char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
             char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);
             char_array_3[2] = ((char_array_4[2] & 0x3) << 6) + char_array_4[3];


             for (i =0; (i < 3); i++) {
                 ret += char_array_3[i];
             }
             i = 0;

         }

     }

     if(i) {
         for (j =0; j < i; j++) {
             char_array_4[j] = this->base64_chars.find(char_array_4[j]);
         }
         char_array_3[0] = (char_array_4[0] << 2) + ((char_array_4[1] & 0x30) >> 4);
         char_array_3[1] = ((char_array_4[1] & 0xf) << 4) + ((char_array_4[2] & 0x3c) >> 2);


         for (j =0; (j < i -1); j++) {
             ret += char_array_3[j];
         }

        
     }
      return ret;

 }

 cv::Mat MtcnnCrop::readBase64Image(std::string encoded_string) {

    std::string decoded_string = this->base64_decode(encoded_string);
    std::vector<uchar> data(decoded_string.begin(), decoded_string.end());


    return imdecode(data, cv::IMREAD_UNCHANGED);

}

/**
 * Decode data to base 64
 * */
std::string MtcnnCrop::base64_encode(unsigned char const* bytes_to_encode, unsigned int in_len) {
  std::string ret;
  int i = 0;
  int j = 0;
  unsigned char char_array_3[3];
  unsigned char char_array_4[4];

  while (in_len--) {
    char_array_3[i++] = *(bytes_to_encode++);
    if (i == 3) {
      char_array_4[0] = (char_array_3[0] & 0xfc) >> 2;
      char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
      char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);
      char_array_4[3] = char_array_3[2] & 0x3f;

      for(i = 0; (i <4) ; i++)
        ret += base64_chars[char_array_4[i]];
      i = 0;
    }
  }

  if (i)
  {
    for(j = i; j < 3; j++)
      char_array_3[j] = '\0';

    char_array_4[0] = ( char_array_3[0] & 0xfc) >> 2;
    char_array_4[1] = ((char_array_3[0] & 0x03) << 4) + ((char_array_3[1] & 0xf0) >> 4);
    char_array_4[2] = ((char_array_3[1] & 0x0f) << 2) + ((char_array_3[2] & 0xc0) >> 6);

    for (j = 0; (j < i + 1); j++)
      ret += base64_chars[char_array_4[j]];

    while((i++ < 3))
      ret += '=';

  }

  return ret;

}

std::string MtcnnCrop::imagetobase64(cv::Mat image){

    std::vector<uchar> buf;
    cv::imencode(".jpg", image, buf);
    auto *enc_msg = reinterpret_cast<unsigned char*>(buf.data());
    std::string encoded = base64_encode(enc_msg, buf.size());
    return encoded;
}