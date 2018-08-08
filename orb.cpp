#include <opencv2\xfeatures2d\nonfree.hpp>
#include <xfeatures2d.hpp>
#include <iostream>
#include <cv.h>
#include <core.hpp>
#include <highgui.h>
#include <opencv.hpp>

using namespace cv;
using namespace std;
int siftTest(string path1, string path2);
int gmsTest(string path1, string path2);
void imresize(Mat&, int);
void gmsPinjie(string path1, string path2, string outpath);
void siftPinjie(string path1, string path2, string outpath);
Point2f getTransformPoint(const Point2f originalPoint, const Mat &transformMaxtri);

int main(int argc, char * argv[]) {
	if (argc<5)
	{
		if (argc > 1) {
			string flag = argv[1];
			if (flag == "test") {
				string path1 = argv[2];
				string path2 = argv[3];
				gmsTest(path1, path2);
				siftTest(path1, path2);
			}
		}
		cout << "****************************************" << endl;
		cout << "** the program is to test gms and sift" << endl;
		cout << "** if you just want to test sift or gms you should input 3 argv" << endl;
		cout << "** 'test' img1_path img2_path" << endl;
		cout << "** if you want to test pinjie" << endl;
		cout << "** you should input 4 argv" << endl;
		cout << "** img1_path img2_path gmsoutpath siftoutpath" << endl;
		system("pause");
	}
	else
	{
		string path1 = argv[1];
		string path2 = argv[2];
		string outpath1 = argv[3];
		string outpath2 = argv[4];
		gmsPinjie(path1, path2, outpath1);
		siftPinjie(path1, path2, outpath2);

		//gmsTest();
		//siftTest();
	}
}

void gmsPinjie(string path1, string path2, string outpath) {

	Mat img1 = imread(path1);
	Mat img1_hd = imread(path1, CV_LOAD_IMAGE_GRAYSCALE);
	Mat img2 = imread(path2);
	Mat img2_hd = imread(path2, CV_LOAD_IMAGE_GRAYSCALE);

	imresize(img1, 480);
	imresize(img1_hd, 480);
	imresize(img2, 480);
	imresize(img2_hd, 480);

	vector<KeyPoint> kp1, kp2;
	Mat des1, des2;
	vector<DMatch> matches, matches_gms;
	Ptr<ORB> orb = ORB::create(10000);
	orb->setFastThreshold(0);

	if (img1.rows * img2.rows > 480 * 640) {
		orb->setMaxFeatures(100000);
		orb->setFastThreshold(5);
	}

	orb->detectAndCompute(img1, noArray(), kp1, des1);
	orb->detectAndCompute(img2, noArray(), kp2, des2);

	cout << des1.size() << "GMS.....SIZE" << kp1.size() << endl;

	BFMatcher matcher(NORM_HAMMING);
	matcher.match(des1, des2, matches);

	cv::xfeatures2d::matchGMS(img1.size(), img2.size(), kp1, kp2, matches, matches_gms);
	sort(matches_gms.begin(), matches_gms.end()); //����������	
												  //��ȡ����ǰN��������ƥ��������
	vector<Point2f> imagePoints1, imagePoints2;
	for (int i = 0; i<10; i++)
	{
		imagePoints1.push_back(kp1[matches_gms[i].queryIdx].pt);
		imagePoints2.push_back(kp2[matches_gms[i].trainIdx].pt);
	}

	//��ȡͼ��1��ͼ��2��ͶӰӳ����󣬳ߴ�Ϊ3*3
	Mat homo = findHomography(imagePoints1, imagePoints2, 8);
	Mat adjustMat = (Mat_<double>(3, 3) << 1.0, 0, img1.cols, 0, 1.0, 0, 0, 0, 1.0);
	Mat adjustHomo = adjustMat*homo;

	//��ȡ��ǿ��Ե���ԭʼͼ��;���任��ͼ���ϵĶ�Ӧλ�ã�����ͼ��ƴ�ӵ�Ķ�λ
	Point2f originalLinkPoint, targetLinkPoint, basedImagePoint;
	originalLinkPoint = kp1[matches_gms[0].queryIdx].pt;
	targetLinkPoint = getTransformPoint(originalLinkPoint, adjustHomo);
	basedImagePoint = kp2[matches_gms[0].trainIdx].pt;

	//ͼ����׼
	Mat imageTransform1;
	warpPerspective(img1, imageTransform1, adjustMat*homo, Size(img2.cols + img1.cols + 110, img2.rows + img1.rows));

	//����ǿƥ��������ص���������ۼӣ����ν��ȶ����ɣ�����ͻ��
	Mat image1Overlap, image2Overlap; //ͼ1��ͼ2���ص�����	
	image1Overlap = imageTransform1(Rect(Point(targetLinkPoint.x - basedImagePoint.x, 0), Point(targetLinkPoint.x, img2.rows)));
	image2Overlap = img2(Rect(0, 0, image1Overlap.cols, image1Overlap.rows));
	Mat image1ROICopy = image1Overlap.clone();  //����һ��ͼ1���ص�����
	for (int i = 0; i<image1Overlap.rows; i++)
	{
		for (int j = 0; j<image1Overlap.cols; j++)
		{
			double weight;
			weight = (double)j / image1Overlap.cols;  //�����ı���ı�ĵ���ϵ��
			image1Overlap.at<Vec3b>(i, j)[0] = (1 - weight)*image1ROICopy.at<Vec3b>(i, j)[0] + weight*image2Overlap.at<Vec3b>(i, j)[0];
			image1Overlap.at<Vec3b>(i, j)[1] = (1 - weight)*image1ROICopy.at<Vec3b>(i, j)[1] + weight*image2Overlap.at<Vec3b>(i, j)[1];
			image1Overlap.at<Vec3b>(i, j)[2] = (1 - weight)*image1ROICopy.at<Vec3b>(i, j)[2] + weight*image2Overlap.at<Vec3b>(i, j)[2];
		}
	}
	Mat ROIMat = img2(Rect(Point(image1Overlap.cols, 0), Point(img2.cols, img2.rows)));	 //ͼ2�в��غϵĲ���
	ROIMat.copyTo(Mat(imageTransform1, Rect(targetLinkPoint.x, 0, ROIMat.cols, img2.rows))); //���غϵĲ���ֱ���ν���ȥ
																							 //namedWindow("ƴ�ӽ��",0);
																							 //imshow("ƴ�ӽ��",imageTransform1);	
	imwrite(outpath, imageTransform1);
	imshow("gmsPinjie", imageTransform1);
	//	waitKey(0);

}

void siftPinjie(string path1, string path2, string outpath) {

	Mat img1 = imread(path1);
	Mat img1_hd = imread(path1, CV_LOAD_IMAGE_GRAYSCALE);
	Mat img2 = imread(path2);
	Mat img2_hd = imread(path2, CV_LOAD_IMAGE_GRAYSCALE);

	imresize(img1, 480);
	imresize(img1_hd, 480);
	imresize(img2, 480);
	imresize(img2_hd, 480);

	vector<KeyPoint> kp1, kp2;
	Mat des1, des2;
	Ptr<Feature2D> sift = xfeatures2d::SIFT::create();
	sift->detectAndCompute(img1_hd, noArray(), kp1, des1);
	sift->detectAndCompute(img2_hd, noArray(), kp2, des2);
	cout << des1.size() << "SIFT.....SIZE" << kp1.size() << endl;

	BFMatcher matcher(NORM_L2);
	vector<DMatch> matches;
	matcher.match(des1, des2, matches);
	sort(matches.begin(), matches.end()); //����������	
										  //��ȡ����ǰN��������ƥ��������
	vector<Point2f> imagePoints1, imagePoints2;
	for (int i = 0; i<10; i++)
	{
		imagePoints1.push_back(kp1[matches[i].queryIdx].pt);
		imagePoints2.push_back(kp2[matches[i].trainIdx].pt);
	}

	//��ȡͼ��1��ͼ��2��ͶӰӳ����󣬳ߴ�Ϊ3*3
	Mat homo = findHomography(imagePoints1, imagePoints2, 8);
	Mat adjustMat = (Mat_<double>(3, 3) << 1.0, 0, img1.cols, 0, 1.0, 0, 0, 0, 1.0);
	Mat adjustHomo = adjustMat*homo;

	//��ȡ��ǿ��Ե���ԭʼͼ��;���任��ͼ���ϵĶ�Ӧλ�ã�����ͼ��ƴ�ӵ�Ķ�λ
	Point2f originalLinkPoint, targetLinkPoint, basedImagePoint;
	originalLinkPoint = kp1[matches[0].queryIdx].pt;
	targetLinkPoint = getTransformPoint(originalLinkPoint, adjustHomo);
	basedImagePoint = kp2[matches[0].trainIdx].pt;

	//ͼ����׼
	Mat imageTransform1;
	warpPerspective(img1, imageTransform1, adjustMat*homo, Size(img2.cols + img1.cols + 110, img2.rows + img1.rows));

	//����ǿƥ��������ص���������ۼӣ����ν��ȶ����ɣ�����ͻ��
	Mat image1Overlap, image2Overlap; //ͼ1��ͼ2���ص�����	
	image1Overlap = imageTransform1(Rect(Point(targetLinkPoint.x - basedImagePoint.x, 0), Point(targetLinkPoint.x, img2.rows)));
	image2Overlap = img2(Rect(0, 0, image1Overlap.cols, image1Overlap.rows));
	Mat image1ROICopy = image1Overlap.clone();  //����һ��ͼ1���ص�����
	for (int i = 0; i<image1Overlap.rows; i++)
	{
		for (int j = 0; j<image1Overlap.cols; j++)
		{
			double weight;
			weight = (double)j / image1Overlap.cols;  //�����ı���ı�ĵ���ϵ��
			image1Overlap.at<Vec3b>(i, j)[0] = (1 - weight)*image1ROICopy.at<Vec3b>(i, j)[0] + weight*image2Overlap.at<Vec3b>(i, j)[0];
			image1Overlap.at<Vec3b>(i, j)[1] = (1 - weight)*image1ROICopy.at<Vec3b>(i, j)[1] + weight*image2Overlap.at<Vec3b>(i, j)[1];
			image1Overlap.at<Vec3b>(i, j)[2] = (1 - weight)*image1ROICopy.at<Vec3b>(i, j)[2] + weight*image2Overlap.at<Vec3b>(i, j)[2];
		}
	}
	Mat ROIMat = img2(Rect(Point(image1Overlap.cols, 0), Point(img2.cols, img2.rows)));	 //ͼ2�в��غϵĲ���
	ROIMat.copyTo(Mat(imageTransform1, Rect(targetLinkPoint.x, 0, ROIMat.cols, img2.rows))); //���غϵĲ���ֱ���ν���ȥ
																							 //namedWindow("ƴ�ӽ��",0);
																							 //imshow("ƴ�ӽ��",imageTransform1);	
	imwrite(outpath, imageTransform1);
	imshow("siftPinjie", imageTransform1);
	waitKey(0);

}

int gmsTest(string path1, string path2) {

	Mat img1 = imread(path1);
	Mat img1_hd = imread(path1, CV_LOAD_IMAGE_GRAYSCALE);
	Mat img2 = imread(path2);
	Mat img2_hd = imread(path2, CV_LOAD_IMAGE_GRAYSCALE);

	imresize(img1, 480);
	imresize(img1_hd, 480);
	imresize(img2, 480);
	imresize(img2_hd, 480);

	vector<KeyPoint> kp1, kp2;
	Mat des1, des2;
	vector<DMatch> matches, matches_gms;
	Ptr<ORB> orb = ORB::create(5000);
	orb->setFastThreshold(0);

	if (img1.rows * img2.rows > 480 * 640) {
		orb->setMaxFeatures(100000);
		orb->setFastThreshold(5);
	}

	orb->detectAndCompute(img1, noArray(), kp1, des1);
	orb->detectAndCompute(img2, noArray(), kp2, des2);

	cout << des1.size() << "GMS.....SIZE" << kp1.size() << endl;

	BFMatcher matcher(NORM_HAMMING);
	matcher.match(des1, des2, matches);

	cv::xfeatures2d::matchGMS(img1.size(), img2.size(), kp1, kp2, matches, matches_gms);
	cout << "gms __________ number of matches " << matches_gms.size() << endl;
	Mat show;
	drawMatches(img1, kp1, img2, kp2, matches_gms, show);
	imshow("GMS", show);
	//waitKey(0);



	//xfeatures2d::FASTForPointSet(img1,);
	//Ptr<Feature2D> sift = xfeatures2d::FASTForPointSet
	return 0;
}

int siftTest(string path1, string path2) {

	//	Mat img1 = imread("d:\\Pictures\\testGMS\\1.jpg");
	//	Mat img1_hd = imread("d:\\Pictures\\testGMS\\1.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	//	Mat img2 = imread("d:\\Pictures\\testGMS\\2.jpg");
	//	Mat img2_hd = imread("d:\\Pictures\\testGMS\\2.jpg", CV_LOAD_IMAGE_GRAYSCALE);

	Mat img1 = imread(path1);
	Mat img1_hd = imread(path1, CV_LOAD_IMAGE_GRAYSCALE);
	Mat img2 = imread(path2);
	Mat img2_hd = imread(path2, CV_LOAD_IMAGE_GRAYSCALE);

	imresize(img1, 480);
	imresize(img1_hd, 480);
	imresize(img2, 480);
	imresize(img2_hd, 480);


	vector<KeyPoint> kp1, kp2;
	Mat des1, des2;
	Ptr<Feature2D> sift = xfeatures2d::SIFT::create();
	sift->detectAndCompute(img1_hd, noArray(), kp1, des1);
	sift->detectAndCompute(img2_hd, noArray(), kp2, des2);
	cout << des1.size() << "SIFT.....SIZE" << kp1.size() << endl;
	/*Mat res1, res2;
	int drawmode = DrawMatchesFlags::DRAW_RICH_KEYPOINTS;
	drawKeypoints(img1_hd, kp1, res1, Scalar::all(-1), drawmode);
	drawKeypoints(img2_hd, kp2, res2, Scalar::all(-1), drawmode);
	cout << "size of des1 is " << kp1.size() << endl;
	cout << "size of des2 is " << kp2.size() << endl;

	//write the size of feture on the picture
	CvFont font;
	double hScale = 1;
	double vScale = 1;
	int lineWidth = 2;
	cvInitFont(&font, CV_FONT_HERSHEY_SIMPLEX | CV_FONT_ITALIC, hScale, vScale, 0, lineWidth);//��ʼ������ д��ͼƬ��

	//cvPoint Ϊ��ʵ�x,y ����
	IplImage * transimg1 = cvCloneImage(&(IplImage)res1);
	IplImage * transimg2 = cvCloneImage(&(IplImage)res2);
	char str1[20], str2[20];
	sprintf_s(str1, "%d", kp1.size());
	sprintf_s(str2, "%d", kp2.size());

	const char * str = str1;
	cvPutText(transimg1, str1, cvPoint(0, 0), &font, CV_RGB(255, 0, 0));
	cvPutText(transimg2, str2, cvPoint(0, 0), &font, CV_RGB(255, 0, 0));

	//��������ʾ
	//cvShowImage("des1", transimg1);
	//cvShowImage("des2", transimg2);*/

	//ƥ�䣿
	BFMatcher matcher(NORM_L2);
	vector<DMatch> matches;
	matcher.match(des1, des2, matches);
	Mat imgMatch;
	drawMatches(img1, kp1, img2, kp2, matches, imgMatch);
	//drawMatches(img1, kp1, img2, kp2, matches, imgMatch,Scalar::all(-1), Scalar::all(-1), vector<char>(), drawmode);
	cout << "sift __________ number of matches " << matches.size() << endl;
	imshow("SIFT", imgMatch);

	//waitKey(0);

	//RANSAC ������ƥ�������� ��Ҫ��Ϊ�������֣�
	//1������matches�����������,������ת��Ϊfloat����
	//2��ʹ����������󷽷� findFundamentalMat,�õ�RansacStatus
	//3������RansacStatus������ƥ��ĵ�Ҳ��RansacStatus[i]=0�ĵ�ɾ��

	//����matches�����������,������ת��Ϊfloat����
	vector<KeyPoint> R_keypoint01, R_keypoint02;
	for (size_t i = 0; i<matches.size(); i++)
	{
		R_keypoint01.push_back(kp1[matches[i].queryIdx]);
		R_keypoint02.push_back(kp2[matches[i].trainIdx]);
		//�����仰����⣺R_keypoint1��Ҫ�洢img01������img02ƥ��������㣬
		//matches�д洢����Щƥ���Ե�img01��img02������ֵ
	}

	//����ת��
	vector<Point2f>p01, p02;
	for (size_t i = 0; i<matches.size(); i++)
	{
		p01.push_back(R_keypoint01[i].pt);
		p02.push_back(R_keypoint02[i].pt);
	}

	//���û��������޳���ƥ���
	vector<uchar> RansacStatus;
	Mat Fundamental = findFundamentalMat(p01, p02, RansacStatus, FM_RANSAC);


	vector<KeyPoint> RR_keypoint01, RR_keypoint02;
	vector<DMatch> RR_matches;            //���¶���RR_keypoint ��RR_matches���洢�µĹؼ����ƥ�����
	int index = 0;
	for (size_t i = 0; i<matches.size(); i++)
	{
		if (RansacStatus[i] != 0)
		{
			RR_keypoint01.push_back(R_keypoint01[i]);
			RR_keypoint02.push_back(R_keypoint02[i]);
			matches[i].queryIdx = index;
			matches[i].trainIdx = index;
			RR_matches.push_back(matches[i]);
			index++;
		}
	}
	Mat img_RR_matches;
	drawMatches(img1, RR_keypoint01, img2, RR_keypoint02, RR_matches, img_RR_matches);
	cout << "sift rascan __________ number of matches " << RR_matches.size() << endl;
	imshow("������ƥ����", img_RR_matches);

	waitKey(0);


	cout << "hello" << endl;
	return 0;
}

void imresize(Mat &src, int height) {
	double ratio = src.rows * 1.0 / height;
	int width = static_cast<int>(src.cols * 1.0 / ratio);
	resize(src, src, Size(width, height));
}

Point2f getTransformPoint(const Point2f originalPoint, const Mat &transformMaxtri)
{
	Mat originelP, targetP;
	originelP = (Mat_<double>(3, 1) << originalPoint.x, originalPoint.y, 1.0);
	targetP = transformMaxtri*originelP;
	float x = targetP.at<double>(0, 0) / targetP.at<double>(2, 0);
	float y = targetP.at<double>(1, 0) / targetP.at<double>(2, 0);
	return Point2f(x, y);
}