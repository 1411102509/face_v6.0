#include <vector>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <string>

#include<opencv2/opencv.hpp>
#include<opencv2/highgui.hpp>
#include<opencv2/imgproc.hpp>
#include<opencv2/core.hpp>

#include "arcsoft_face_sdk.h"
#include "amcomdef.h"
#include "asvloffscreen.h"
#include "merror.h"

using namespace std;
using namespace cv;

#define NSCALE 16   //"取值范围[2,32]，VIDEO模式推荐值16，IMAGE模式内部设置固定值为30" 
#define FACENUM 100 //"检测的人脸数"
#define MINI_DETECT_DISTANCE 0.3    //最小识别距离
#define MINI_CONFIDENCE_LEVEL 0.75    //最小判别相似度


typedef struct{
    int FACE_Label;
    string FACE_Name;
    ASF_FaceFeature FACE_Feature;
}FACE_SingleData,*LPFACE_SingleData;

class FaceRec{
    public:
        FaceRec(){
            read_Data_From_Xml();
        }
        
        cv::Point3f get_Ang(int center_x,int center_y,int width);
        
        bool get_Multi_Face_Ang(ASF_MultiFaceInfo &detectedFaces,std::vector<cv::Point3f> &faceAngle,float &minDis);
        
        bool new_Face();
        
        int faces_Db_Comparsion(ASF_FaceFeature &feature);

        int recongnise_Face(bool loop_flag);
        
        bool init_Engine();
        
        bool save_Data_To_Xml();
        
        bool read_Data_From_Xml();
        
        void mark_faces(cv::Mat &img,ASF_MultiFaceInfo &facesPosData,std::vector<cv::Point3f> facesAngle,cv::Scalar color);
        
        void printAllDatas();
        
        int size(){
            return FACES_StoredData.size()+1;
        }
        
        MHandle handle = NULL;
        MRESULT res;
    private:
        std::vector<FACE_SingleData> FACES_StoredData; 

};
