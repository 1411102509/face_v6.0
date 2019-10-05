cmake_minimum_required(VERSION 2.0)		#指定cmake版本 
project(face_6.0)				#指定项目的名称，一般和项目的文件夹名称对应

set(CMAKE_CXX_STANDARD 11)				#设置c++标准

#指定头文件目录
include_directories(./)	
include_directories(./inc)

#指定静态和动态文件目录
link_directories(../linux_so)
#
find_package(OpenCV REQUIRED)

#该指令的主要作用就是将指定的源文件生成链接文件，然后添加到工程中去。
#arcsoft_face生成的库文件名		[STATIC | SHARED | MODULE] 库类型
add_executable(face_6.0
        ./inc/amcomdef.h
        ./inc/arcsoft_face_sdk.h
		./inc/asvloffscreen.h
		./inc/merror.h
		./master.cpp
		./face_Factory.cpp
		./face_Factory.hpp)

#在给定的作用域内设置一个命名的属性
set_property(TARGET face_6.0 PROPERTY POSITION_INDEPENDENT_CODE ON)


#链接库文件
target_link_libraries(face_6.0	
		arcsoft_face
		arcsoft_face_engine
		)

target_link_libraries(face_6.0 ${OpenCV_LIBS})