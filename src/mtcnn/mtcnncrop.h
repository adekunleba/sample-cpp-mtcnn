#ifndef MTCNNCROP_H_
#define MTCNNCROP_H_

#include <opencv2/core/core.hpp>

class MtcnnCrop {


    public:
        MtcnnCrop();
        ~MtcnnCrop();
        cv::Mat cropface(cv::Mat image);
        cv::Mat returnbase64(cv::Mat image);


};


#endif