#include "face_Factory.hpp"

// #define ACTIVATE_SDK
// #define APPID "HV7iMti7TD73cKwPvqVnVoj8FPctY8y9wJNaQxmFUCTG"
// #define SDKKEY "GrZv23NEm1LeNr5W5QpX9dZzSeavGwMhA4W4B92DeYaw"

#define SHOW_IN_DEBUG

cv::Point3f FaceRec::get_Ang(int center_x,int center_y,int width){
	double calibration[9] = {
		945.2654, 0.000000, 701.7872,
		0.000000, 945.6360, 319.5665,
		0.000000, 0.000000, 1.000000
	};
	double dist_coeffs[5] = { -0.0112, 0.0878, 0.0000, 0.0000, -0.0992 };
	Mat cameraMatrix = Mat(3, 3, CV_64F, calibration);	//内参矩阵
	Mat distCoeffs = Mat(5, 1, CV_64F, dist_coeffs);	//畸变系数
	
	vector<Point3f> object_point;
	object_point.push_back(Point3f(-80, -80, 0.0));	//人脸宽80，以中心为原点
	object_point.push_back(Point3f(80,  -80, 0.0));
	object_point.push_back(Point3f(80,  80 , 0.0));
	object_point.push_back(Point3f(-80, 80 , 0.0));

	vector<Point2f> image_point;
	image_point.push_back(Point2f(center_x - width, center_y - width));
	image_point.push_back(Point2f(center_x + width, center_y - width));
	image_point.push_back(Point2f(center_x + width, center_y + width));
	image_point.push_back(Point2f(center_x - width, center_y + width));

	Mat rvec = Mat::ones(3, 1, CV_64F);	//旋转矩阵
	Mat tvec = Mat::ones(3, 1, CV_64F);	//平移矩阵

	solvePnP(object_point, image_point, cameraMatrix, distCoeffs, rvec, tvec);
	
	double pos_x, pos_y, pos_z;
	const double *_xyz = (const double *)tvec.data;
	pos_z = tvec.at<double>(2) / 1000.0;
	pos_x = atan2(_xyz[0], _xyz[2]);
	pos_y = atan2(_xyz[1], _xyz[2]);
	pos_x *= 180 / 3.1415926;
	pos_y *= 180 / 3.1415926;

	Point3f pos = Point3f(pos_x,pos_y,pos_z);
	return pos;
}

bool FaceRec::get_Multi_Face_Ang(ASF_MultiFaceInfo &detectedFaces,std::vector<cv::Point3f> &faceAngle,float &minDis){
	minDis = 1024;
	for(int i = 0;i<detectedFaces.faceNum;i++){
		//计算出每个人脸中心坐标
		int center_x,center_y,width;
		center_x = (detectedFaces.faceRect[i].left + detectedFaces.faceRect[i].right) / 2;
		center_y = (detectedFaces.faceRect[i].bottom + detectedFaces.faceRect[i].top) / 2;
		width = detectedFaces.faceRect[i].right - detectedFaces.faceRect[i].left;

		faceAngle.push_back(get_Ang(center_x,center_y,width));
		if(faceAngle[i].z < minDis)
			minDis = faceAngle[i].z;
	}
	return 1;
}


void FaceRec::printAllDatas(){
	cout<<"Perpare to print all datas..."<<endl;
	for(int i = 0;i<FACES_StoredData.size();i++){
		cout<<"*************************"<<i<<"************************"<<endl;
		cout<<"FACE_Label: "<<FACES_StoredData[i].FACE_Label<<endl;
		cout<<"FACE_Name: "<<FACES_StoredData[i].FACE_Name<<endl;
		cout<<"FACE_Feature: ";
		for(int j = 0;j<FACES_StoredData[i].FACE_Feature.featureSize;j++){
			cout<<(int)FACES_StoredData[i].FACE_Feature.feature[j]<<" ";
		}
		cout<<endl;
	}
	cout<<"Print all datas complete!"<<endl;
}

void FaceRec::mark_faces(cv::Mat &img,ASF_MultiFaceInfo &facesPosData,std::vector<cv::Point3f> facesAngle,cv::Scalar color){
	cv::Rect _position;
    for(int i = 0;i<facesPosData.faceNum;i++){
        string text = cv::format("(%0.2f,%0.2f,%0.2f)",facesAngle[i].x,facesAngle[i].y,facesAngle[i].z);
        putText(img,text,Point(facesPosData.faceRect[i].left,facesPosData.faceRect[i].top - 5),FONT_HERSHEY_PLAIN,1,color);//显示解算角度

		_position.x = facesPosData.faceRect[i].left;
		_position.y = facesPosData.faceRect[i].top;
		_position.height = facesPosData.faceRect[i].bottom - facesPosData.faceRect[i].top;
		_position.width = facesPosData.faceRect[i].right - facesPosData.faceRect[i].left;

        rectangle(img,_position,color,2,8);
    }
}

bool FaceRec::save_Data_To_Xml(){
	cout<<"Perpare to save "<<FaceRec::FACES_StoredData.size()<<" faces..."<<endl;
    cv::FileStorage fs("../Faces_DB/test.xml",cv::FileStorage::WRITE); //写操作
	fs<<"FACES_StoredData"<<"[";
	for(int i = 0;i<FACES_StoredData.size();i++){
    	fs<<"{";
			fs<<"FACE_Label"<<FACES_StoredData[i].FACE_Label;
			fs<<"FACE_Name"<<FACES_StoredData[i].FACE_Name;
			fs<<"Feature_Size"<<FACES_StoredData[i].FACE_Feature.featureSize;
			fs<<"Features"<<"[";
				for(int j = 0;j<FACES_StoredData[i].FACE_Feature.featureSize;j++){
					fs<<(int)FACES_StoredData[i].FACE_Feature.feature[j];
				}
			fs<<"]";
		fs<<"}";
	}
	fs<<"]";
	cout<<"Save complete!"<<endl;
}

bool FaceRec::read_Data_From_Xml(){
	cout<<"Perpare to read faces_datas..."<<endl;
	FACES_StoredData.clear();
	FACE_SingleData tempFaceData;					//临时保存从xml读取到的一个人脸数据

    cv::FileStorage fs("../Faces_DB/test.xml",cv::FileStorage::READ); //读操作

	FileNode faces_node = fs["FACES_StoredData"];	//读取根节点
    FileNodeIterator fni = faces_node.begin(); 		//获取结构体数组迭代器
    FileNodeIterator fniEnd = faces_node.end();
    for(;fni != fniEnd;fni++){	//遍历所有人脸数据
        tempFaceData.FACE_Label = (int)(*fni)["FACE_Label"];
        tempFaceData.FACE_Name = (string)(*fni)["FACE_Name"];
        tempFaceData.FACE_Feature.featureSize = (int)(*fni)["Feature_Size"];

		//读取人脸特征
        FileNode features = (*fni)["Features"];
        FileNodeIterator fni2 = features.begin(); 	//获取结构体数组迭代器
        FileNodeIterator fniEnd2 = features.end();
		tempFaceData.FACE_Feature.feature = new MByte[tempFaceData.FACE_Feature.featureSize];//开辟空间存储feature
        for(int count2 = 0;fni2 != fniEnd2;fni2++,count2++){	//遍历所有feature数据
			tempFaceData.FACE_Feature.feature[count2] = (int)(*fni2);
        }
		FACES_StoredData.push_back(tempFaceData);
    }
	cout<<"Read complete,now faceNum = "<<FACES_StoredData.size()<<endl;
	// delete []tempFaceData.FACE_Feature.feature;	//释放内存，但会出现问题
	return 1;
}

bool FaceRec::init_Engine(){
    MRESULT res;
	//激活SDK
    #ifdef ACTIVATE_SDK
        res = ASFOnlineActivation(APPID, SDKKEY);
        if (MOK != res && MERR_ASF_ALREADY_ACTIVATED != res)
            #ifdef SHOW_IN_DEBUG
                printf("ASFOnlineActivation fail: %d\n", res);
            #endif
        else
            #ifdef SHOW_IN_DEBUG
                printf("ASFOnlineActivation sucess: %d\n", res);
            #endif
    #endif
	//初始化引擎
    
	handle = NULL;	//引擎句柄
	MInt32 mask = ASF_FACE_DETECT | ASF_FACERECOGNITION | ASF_AGE | ASF_GENDER | ASF_FACE3DANGLE | ASF_LIVENESS | ASF_IR_LIVENESS;	//需要启用的功能组合
	res = ASFInitEngine(ASF_DETECT_MODE_IMAGE, ASF_OP_0_ONLY, NSCALE, FACENUM, mask, &handle);	//初始化引擎
	if (res != MOK){
        #ifdef SHOW_IN_DEBUG
		    printf("ALInitEngine fail: %d\n", res);
        #endif
        return 0;
    }
	else{
        #ifdef SHOW_IN_DEBUG
		    printf("ALInitEngine sucess: %d\n", res);
        #endif
        return 1;
    }
}

bool FaceRec::new_Face(){
	cout<<"Perpare to new face..."<<endl;
	cv::VideoCapture cap(0);
	cv::Mat frame;
	while(1){
		cap >> frame;
		ASF_MultiFaceInfo detectedFaces = { 0 };
		ASF_SingleFaceInfo SingleDetectedFaces = { 0 };
		ASF_FaceFeature feature = { 0 };
		FACE_SingleData tempFaceData = { 0 };
		res = ASFDetectFaces(handle, frame.cols, frame.rows, ASVL_PAF_RGB24_B8G8R8, (MUInt8*)frame.data, &detectedFaces);
		if(res != MOK){
			cout<<"detect error:"<<(int)res<<endl;
		}

		std::vector<cv::Point3f> faceAngle; //人脸在世界坐标的位置
		float minDis;                //在摄像头前的人脸最小距离
		FaceRec::get_Multi_Face_Ang(detectedFaces,faceAngle,minDis);
		cout<<"MiniDis:"<<minDis<<endl;

		if(minDis<MINI_DETECT_DISTANCE){
			int minLabel = 0;
			for(int i = 0;i<faceAngle.size();i++){
				if(faceAngle[i].z == minDis){
					minLabel = i;
					break;
				}
			}
			mark_faces(frame,detectedFaces,faceAngle,cv::Scalar(0,0,255));
			//人脸特征提取
			SingleDetectedFaces.faceRect.left = detectedFaces.faceRect[minLabel].left;
			SingleDetectedFaces.faceRect.top = detectedFaces.faceRect[minLabel].top;
			SingleDetectedFaces.faceRect.right = detectedFaces.faceRect[minLabel].right;
			SingleDetectedFaces.faceRect.bottom = detectedFaces.faceRect[minLabel].bottom;
			SingleDetectedFaces.faceOrient = detectedFaces.faceOrient[minLabel];

			res = ASFFaceFeatureExtract(handle, frame.cols, frame.rows, ASVL_PAF_RGB24_B8G8R8, (MUInt8*)frame.data, &SingleDetectedFaces, &feature);
			if (res == MOK){
				cout<<"原始数据面部特征提取成功，返回码: "<<(int)res<<endl;
				//拷贝feature
				tempFaceData.FACE_Feature.featureSize = feature.featureSize;
				tempFaceData.FACE_Feature.feature = (MByte *)malloc(feature.featureSize);
				memset(tempFaceData.FACE_Feature.feature, 0, feature.featureSize);
				memcpy(tempFaceData.FACE_Feature.feature, feature.feature, feature.featureSize);
				tempFaceData.FACE_Label = FACES_StoredData.size();
				tempFaceData.FACE_Name = "未命名";

				FaceRec::FACES_StoredData.push_back(tempFaceData);
				save_Data_To_Xml();
				string path = cv::format("../Faces_DB/%d_%s.jpg",tempFaceData.FACE_Label,tempFaceData.FACE_Name.c_str());
				cv::imwrite(path,frame);
				cout<<"New face complete!"<<endl;
				return 1;
			}
			else 
				cout<<"原始数据面部特征提取失败，返回码: "<<(int)res<<endl;
		}
		else
			mark_faces(frame,detectedFaces,faceAngle,cv::Scalar(0,255,0));
		cv::imshow("detect",frame);
		char c = cv::waitKey(10);
		if(c==27)
			return 0;
	}
}

int FaceRec::faces_Db_Comparsion(ASF_FaceFeature &feature){
	int resultLabel = 0;
	MFloat resultConfidenceLevel = 0;
	//遍历对比所有人脸数据集
	for(int i = 0;i<FACES_StoredData.size();i++){
		MFloat confidenceLevel;
		//人脸1:1对比
		res = ASFFaceFeatureCompare(handle, &feature, &FACES_StoredData[i].FACE_Feature, &confidenceLevel);
		if (res != MOK){
			cout<<"人脸特征对比失败，返回码: "<<(int)res<<endl;
			return -1024;
		}
		else{
			if(confidenceLevel>resultConfidenceLevel){
				resultConfidenceLevel = confidenceLevel;
				resultLabel = i;
			}
		}
	}

	// cout<<"Result_Label:"<<resultLabel<<" with resultConfidenceLevel:"<<resultConfidenceLevel<<endl;
	if(resultConfidenceLevel>MINI_CONFIDENCE_LEVEL)	//遍历到，返回最可能人脸标签
		return FACES_StoredData[resultLabel].FACE_Label;
	else	//没有遍历到，返回-1，说明为陌生人
		return -1;
}

int FaceRec::recongnise_Face(bool loop_flag = 1){
	cv::VideoCapture cap(0);
	cv::Mat frame;
	while(1){
		cap >> frame;

		ASF_MultiFaceInfo detectedFaces = { 0 };//多人脸信息；
		ASF_SingleFaceInfo SingleDetectedFaces1 = { 0 };
		ASF_FaceFeature feature1 = { 0 };

		//人脸检测
		res = ASFDetectFaces(handle, frame.cols, frame.rows, ASVL_PAF_RGB24_B8G8R8, (MUInt8*)frame.data, &detectedFaces);
		if (MOK == res){
			//进行PNP解算,找出每张脸在世界坐标的位置
			std::vector<cv::Point3f> faceAngle; //人脸在世界坐标的位置
			float minDis;                //在摄像头前的人脸最小距离
			FaceRec::get_Multi_Face_Ang(detectedFaces,faceAngle,minDis);
			cout<<"MiniDis:"<<minDis<<endl;

			if(minDis<MINI_DETECT_DISTANCE){ 
				//标注人脸
				FaceRec::mark_faces(frame,detectedFaces,faceAngle,cv::Scalar(0,0,255));
				
				// 人脸信息检测
				MInt32 processMask = ASF_AGE | ASF_GENDER | ASF_FACE3DANGLE;
				res = ASFProcess(handle, frame.cols, frame.rows, ASVL_PAF_RGB24_B8G8R8, (MUInt8*)frame.data, &detectedFaces, processMask);
				if (res != MOK)
					cout<<"img2人脸信息检测失败，返回码: " << (int)res;
				else
					cout<<"img2人脸信息检测成功，返回码: " << (int)res; 
				
				//获取活体信息
				ASF_LivenessInfo rgbLivenessInfo = { 0 };
				res = ASFGetLivenessScore(handle, &rgbLivenessInfo);
				if (res != MOK)
					cout<<"ASFGetLivenessScore fail: "<<(int)res<<endl;
				else
					printf("ASFGetLivenessScore sucess: %d\n", rgbLivenessInfo.isLive[0]);

				//人脸特征提取
				SingleDetectedFaces1.faceRect.left = detectedFaces.faceRect[0].left;
				SingleDetectedFaces1.faceRect.top = detectedFaces.faceRect[0].top;
				SingleDetectedFaces1.faceRect.right = detectedFaces.faceRect[0].right;
				SingleDetectedFaces1.faceRect.bottom = detectedFaces.faceRect[0].bottom;
				SingleDetectedFaces1.faceOrient = detectedFaces.faceOrient[0];
				res = ASFFaceFeatureExtract(handle, frame.cols, frame.rows, ASVL_PAF_RGB24_B8G8R8, (MUInt8*)frame.data, &SingleDetectedFaces1, &feature1);
				if (res == MOK){
					//与数据库特征对比
					MFloat compareResult = faces_Db_Comparsion(feature1);
					cout<<"对比结果: "<<compareResult<<endl;
				}
				else 
					cout<<"面部特征提取失败，返回码: "<<(int)res<<endl;
			}
			else
				mark_faces(frame,detectedFaces,faceAngle,cv::Scalar(0,255,0));
			
			
		}
		else
			cout<<"面部检测失败，返回码"<<(int)res<<endl;

		cv::imshow("faceRecongnise",frame);
		char c = cv::waitKey(10);
		if(c==27)
			break;
	}
}