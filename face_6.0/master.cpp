#include "face_Factory.hpp"
#include <iostream>  
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;


#define SafeFree(p) { if ((p)) free(p); (p) = NULL; }
#define SafeArrayDelete(p) { if ((p)) delete [] (p); (p) = NULL; } 
#define SafeDelete(p) { if ((p)) delete (p); (p) = NULL; } 
	
int main(){
	FaceRec _face;
	_face.init_Engine();
	_face.new_Face();
	_face.recongnise_Face(1);
}
