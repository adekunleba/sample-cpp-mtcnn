#include <iostream>
#include "src/mtcnncrop.h"

int main(){
    std::string image = "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";



    // ProposalNetwork::Config pConfig;
    // pConfig.caffeModel = "models/det1.caffemodel";
    // pConfig.protoText = "models/det1.prototxt";
    // pConfig.threshold = 0.6f;

    // RefineNetwork::Config rConfig;
    // rConfig.caffeModel = "models/det2.caffemodel";
    // rConfig.protoText = "models/det2.prototxt";
    // rConfig.threshold = 0.7f;

    // OutputNetwork::Config oConfig;
    // oConfig.caffeModel = "models/det3.caffemodel";
    // oConfig.protoText = "models/det3.prototxt";
    // oConfig.threshold = 0.7f;

    ModelPaths det1 = {"models/det1.caffemodel", "models/det1.prototxt"};
    ModelPaths det2 = {"models/det2.caffemodel", "models/det2.prototxt"};
    ModelPaths det3 = {"models/det3.caffemodel", "models/det3.prototxt"};


    MtcnnCrop mtcnn(det1, det2, det3);

    cv::Mat imagemat = mtcnn.readBase64Image(image);


    cv::Mat croppedFace = mtcnn.cropface(imagemat);


    //Convert cropped face to base64
    std::string decodedface = mtcnn.imagetobase64(croppedFace);

    std::cout << "Decoded image base64 \n" << decodedface << std::endl;

    


    return -1;
}
