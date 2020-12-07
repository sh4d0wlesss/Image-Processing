#include<opencv2/opencv.hpp>
#include<stdio.h>
#include<stdlib.h>
#include<time.h>

using namespace cv;
using namespace std;

unsigned char** randomCreator(int, Mat);
float dist(unsigned char, unsigned char, unsigned char, unsigned char, unsigned char, unsigned char);
unsigned char **clustring(Mat, int, int, unsigned char**);
void printMatrix(int, int, unsigned char**);
Mat segmentation(Mat,Mat, unsigned char**);
int get_equal(int**, int);
void add_equal(int**, int, int);
Mat k_means_result(Mat, unsigned char**, unsigned char**);

int main() {

	srand(time(NULL));
	//#######################################################################################################
	
	// KLASÖRE İŞLENECEK RESMİ image.jpg OLARAK VERİNİZ!!!

	Mat img = imread("image.jpg", IMREAD_COLOR);
	Mat modifiedImg = imread("image.jpg", IMREAD_COLOR);
	Mat finalimg = imread("image.jpg", IMREAD_COLOR);
	
	int k;
	printf("k degeri giriniz = ");
	scanf("%d", &k);
	unsigned char** labelmat;
	labelmat = (unsigned char**)malloc(sizeof(unsigned char*) * img.rows);
	for (int i = 0; i < img.rows; i++) {
		labelmat[i] = (unsigned char*)malloc(sizeof(unsigned char) * img.cols);
	}

	//=======================================================
	unsigned char** randomPixel = randomCreator(k, img);
	labelmat = clustring(img, k, 100,randomPixel);
	//=======================================================
	modifiedImg=k_means_result(modifiedImg, labelmat, randomPixel);
	
	 finalimg=segmentation(img,finalimg, labelmat);

	imshow("img", img);
	imshow("modified", modifiedImg);
	imshow("finalimg", finalimg);
	waitKey(0);

	return 0;
}

unsigned char** randomCreator(int k, Mat img) { // this function create a "mü" matrix by selecting a random pixels, rows are r,g,b and cols are value of them

	unsigned char** randArray = (unsigned char**)malloc(sizeof(unsigned char) * 3);
	for (int i = 0; i < 3; i++) {
		randArray[i] = (unsigned char*)malloc(sizeof(unsigned char) * k);
	}

	int cols = img.cols;
	int rows = img.rows;

	unsigned char random_Row;
	unsigned char random_Col;

	for (int i = 0; i < k; i++) {
		random_Row = rand() % rows;
		random_Col = rand() % cols;
		randArray[0][i] = img.at<Vec3b>(random_Row, random_Col)[0]; // blue value
		randArray[1][i] = img.at<Vec3b>(random_Row, random_Col)[1]; // green value
		randArray[2][i] = img.at<Vec3b>(random_Row, random_Col)[2]; // red value
	}
	return randArray;
}

float dist(unsigned char a1, unsigned char b1, unsigned char c1, unsigned char a2, unsigned char b2, unsigned char c2) {// find distance between to pixel 
	return (float)sqrt((a1 - a2) * (a1 - a2) + (b1 - b2) * (b1 - b2) + (c1 - c2) * (c1 - c2));
}

unsigned char **clustring(Mat img, int k, int treshold,unsigned char** randomPixel) {

	
	int control = treshold + 1; // en az 1 kere isleme girmesini istedigimiz için tresholddan büyük bir deger verdik.
	int minDist; // o anki pixelin hangi Mü degerine yakin oldugunu bulmak için kullandigimiz degisken.
	unsigned char minDistIndex; // anlik pixelin en yakın oldugu Mü degerinin indisi.
	int distance; // anlık uzaklık
	int *count;
	count = (int*)malloc(sizeof(int) * k); // her Mü deðeri için kaç adet yakinsayan pixel oldugunu tutan sayac.
	//----------------------------------------------------------------------------------------------------------------------
	// her mü degerine yakınsayan pixelerin r g b degerlerinin toplamının tutldugu matris
	int** sum = (int**) malloc(sizeof(int*) * 3); // 1. indis mavi - 2. indis yesil - 3. indis kirmizi toplamlari tutacak.
	for (int i = 0; i < 3; i++)
		sum[i] = (int*)malloc(sizeof(int) * k);
	//----------------------------------------------------------------------------------------------------------------------
	// label matrisi olusturulmasi---// her pixelin hangi mü degerine en yakin oldugunun tutuldugu matris
	unsigned char** labelMat;
	labelMat = (unsigned char**)malloc(sizeof(unsigned char*) * img.rows);
	for (int i = 0; i < img.rows; i++) {
		labelMat[i] = (unsigned char*)malloc(sizeof(unsigned char) * img.cols);
	} 
	//----------------------------------------------------------------------------------------------------------------------

	while (treshold < control) {

		for (int i = 0; i < k; i++) { // mü degerine yakınsayan degerlerin toplamlarının tutuldugu matris sıfırlanır
			sum[0][i] = 0;
			sum[1][i] = 0;
			sum[2][i] = 0;
			count[i] = 0;
		}
		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				// ilk olarak minimum distanceye ilk deger ataması yapılır, bunu yaparken de ilk mü degerini seçtik
				minDist = dist(img.at<Vec3b>(i, j)[0], img.at<Vec3b>(i, j)[1], img.at<Vec3b>(i, j)[2], 
					randomPixel[0][0], randomPixel[1][0], randomPixel[2][0]);
				
				minDistIndex = 0;// indexini de sıfır yaptık

				for (int m = 1; m < k; m++) {// bu dongu ile pixelin her mü elemaniyla olan uzakligini bulup minimum olani seciyoruz
					
					distance = dist(img.at<Vec3b>(i, j)[0], img.at<Vec3b>(i, j)[1], img.at<Vec3b>(i, j)[2],
						randomPixel[0][m], randomPixel[1][m], randomPixel[2][m]);
					
						if (distance < minDist) {

							minDist = distance;
							minDistIndex = m;
					}
				}
				// en yakın olan mü degerini bulduktan sonra o pixeldeki degerleri mü toplamlarının oldugu yere atıyoruz cünkü daha sonra ortalamalarini alicaz
				labelMat[i][j] = minDistIndex;
				sum[0][minDistIndex] += img.at<Vec3b>(i, j)[0];
				sum[1][minDistIndex] += img.at<Vec3b>(i, j)[1];
				sum[2][minDistIndex] += img.at<Vec3b>(i, j)[2];
				count[minDistIndex]++;

			}
		}

		control = 0;
		for (int i = 0; i < k; i++) {// mü degerlerinin eski degerleri ile arasındaki farkları toplayıp threshold degerini asip asmadigini bulmak icin hesapliyoruz
			sum[0][i] = sum[0][i] / count[i];
			sum[1][i] = sum[1][i] / count[i];
			sum[2][i] = sum[2][i] / count[i];
			// mü degerlerinin eski degerleri ile arasındaki farkları toplayıp threshold degerini asip asmadigini bulmak icin hesapliyoruz
			control += dist(randomPixel[0][i], randomPixel[1][i], randomPixel[2][i], sum[0][i], sum[1][i], sum[2][i]);
		}

		for (int i = 0; i < k; i++) {
			// mü degerlerine yeni deger olarak ortalamalarini atiyoruz
			randomPixel[0][i] = sum[0][i];
			randomPixel[1][i] = sum[1][i];
			randomPixel[2][i] = sum[2][i];
		}
	}
	//dönüs olarak hangi pixele hangi mü degerinin yakin oldugunu tuttugumuz matrisi döndürüyoruz
	return labelMat;
}

void printMatrix(int rows, int cols, unsigned char** mat) { // print the matrix
	printf("\nMATRIX:\n");
	for (int i = 0; i < rows; i++) {
		for (int j = 0; j < cols; j++) {
			printf("%d ", mat[i][j]);
		}
		printf("\n");
	}
}


void add_equal(int** relation_mat, int max, int min) {
	static int count=0;
	if (relation_mat[count] == NULL) {
		printf("error");
		exit(0);
	}
	relation_mat[count][0] = max;
	relation_mat[count][1] = min;
	count++;
}

int get_equal(int** relation_mat, int label) {
	int i = 0;

	while (relation_mat[i][0] != label) {
		i++;
	}
	return relation_mat[i][1];
}


Mat segmentation(Mat img, Mat finalimg, unsigned char** labelMat) {// segmentMat= yeni segmentlerin tutlacagi matris, labelmat= yakinsanmis olan mü degerlerinin tutuldugu matris
	int i;
	int j;
	int labelcount = 0;// label numarasi olarak segmenmatta verdigimiz degerler
	//=======================================================================RELATION MATRIS VE SEGMENT MATRİS TANIMLA
	int** relation = (int**)malloc(sizeof(int*) * (img.rows*img.cols));
	for (i = 0; i < img.cols*img.rows; i++) {
		relation[i] = (int*)malloc(sizeof(int) * 2);
	}
	
	int** segmentMat;
	segmentMat = (int**)malloc(sizeof(int*) * img.rows);
	for (int i = 0; i < img.rows; i++) {
		segmentMat[i] = (int*)malloc(sizeof(int) * img.cols);
	}
	//=======================================================================

	segmentMat[0][0] = 0;// ilk pixele ilk labeli verdik
	add_equal(relation, 0, 0);

	//======================================================================= İLK SATIR VE İLK SÜTUN DOLAŞ
	for (i = 1; i < img.cols; i++)// ilk satırı dolaş
	{
		if (labelMat[0][i] == labelMat[0][i - 1]) {
			segmentMat[0][i] = segmentMat[0][i - 1];
		}
		else {
			labelcount++;
			segmentMat[0][i] = labelcount;
			add_equal(relation, labelcount, labelcount);
		}
	}
	for (i = 1; i < img.rows; i++)// ilk sütunu dolaş
	{
		if (labelMat[i][0] == labelMat[i - 1][0]) {
			segmentMat[i][0] = segmentMat[i - 1][0];
		}
		else {
			labelcount++;
			segmentMat[i][0] = labelcount;
			add_equal(relation, labelcount, labelcount);
		}
	}
	//=======================================================================


	for (i = 1; i < img.rows; i++)
	{
		for (j = 1; j < img.cols; j++)
		{
			
			if (labelMat[i][j] == labelMat[i - 1][j]) {// üstteki satirla labeli ayni mi
				segmentMat[i][j] = get_equal(relation, segmentMat[i-1][j]);
			}
			else if (labelMat[i][j] == labelMat[i][j - 1]) {// solundaki ile labeli ayni mi
				segmentMat[i][j] = get_equal(relation, segmentMat[i][j - 1]);
			}
			else if (labelMat[i][j] == labelMat[i - 1][j - 1]) {// sol üst çaprazindaki ile labeli ayni mi
				segmentMat[i][j] = get_equal(relation, segmentMat[i-1][j - 1]);
			}
			else {
				labelcount++;
				segmentMat[i][j] = labelcount;
				add_equal(relation, labelcount, labelcount);
			}
			if (labelMat[i - 1][j] == labelMat[i][j - 1] && segmentMat[i - 1][j] != segmentMat[i][j - 1]) {// solu ile üstünün labellari ayni ama segmentleri farkli ise
				if ( get_equal(relation,segmentMat[i][j-1]) < get_equal(relation, segmentMat[i - 1][j])) {
					relation[segmentMat[i - 1][j]][1] = get_equal(relation,segmentMat[i][j-1]);
				}
				else {
					relation[segmentMat[i][j - 1]][1] = get_equal(relation, segmentMat[i-1][j]);
				}
			}
		}
	}
	
	for (i = 0; i < img.rows; i++) {
		for ( j = 0; j < img.cols; j++){
			int finall = get_equal(relation, segmentMat[i][j]);

			finalimg.at<Vec3b>(i, j)[0] = (finall * 50) % 255;
			finalimg.at<Vec3b>(i, j)[1] = (finall * 75) % 255;
			finalimg.at<Vec3b>(i, j)[2] = (finall * 100) % 255;
		}
	}


	return finalimg;

}

Mat k_means_result(Mat img, unsigned char** labelmat,unsigned char ** randomPixel) {
	int i = 0,j = 0;
	int index = 0;

	for (i = 0; i < img.rows; i++) {
		for ( j = 0; j < img.cols; j++){
			index = labelmat[i][j];
			img.at<Vec3b>(i, j)[0] = randomPixel[0][index];
			img.at<Vec3b>(i, j)[1] = randomPixel[1][index];
			img.at<Vec3b>(i, j)[2] = randomPixel[2][index];

		}
	}
	return img;
}
