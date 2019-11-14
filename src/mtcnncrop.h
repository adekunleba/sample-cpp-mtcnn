#ifndef MTCNNCROP_H_
#define MTCNNCROP_H_

#include <opencv2/core/core.hpp>
#include "mtcnn/mtcnn/detector.h"

struct ModelPaths {
    std::string detcaffepath;
    std::string detprototxtpath;
};


class MtcnnCrop {

    private:
        std::unique_ptr<MTCNNDetector> _mtcnndetector;
        static const std::string base64_chars;



    public:
        MtcnnCrop();
        MtcnnCrop(ModelPaths det1, ModelPaths det2, ModelPaths det3);
        ~MtcnnCrop();

        cv::Mat cropface(cv::Mat image);
        cv::Mat readBase64Image(std::string encoded_string);
        std::string imagetobase64(cv::Mat image);

        //cv::Mat returnbase64(cv::Mat image);

    private:
        cv::Mat cropwithrect(const cv::Mat image, cv::Rect rect);
        bool is_base64(unsigned char c);
        std::string base64_decode(std::string const & encoded_string);
        std::string base64_encode(unsigned char const* bytes_to_encode, unsigned int in_len);

        
        



};


#endif