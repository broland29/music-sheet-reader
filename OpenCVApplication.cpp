#include "stdafx.h"
#include "common.h"


#define IMAGE_PATH "Images/tannenbaum.bmp"				// path of image being processed
#define THRESHOLD_FOR_BINARY 150						// below this, consider black (0, object) pixel, above white (1, background) pixel
#define THRESHOLD_FOR_LINE 0.5							// a row is considered to have a music sheet line if it has more than this percent (of the image width) pixels black

#define SHOW_OPEN_GRAYSCALE_IMAGE false
#define SHOW_CONVERT_TO_BINARY true
#define SHOW_HORIZONTAL_PROJECTION false
#define SHOW_EXTRACT_LINES false
#define SHOW_DILATION false
#define SHOW_EROSION false
#define SHOW_OPENING true


uchar pattern[9] = {
	255,   0,	255,
	  0,   0,	  0,
	255,   0,	255
};
const Mat_<uchar> sel = Mat(3, 3, CV_8UC1, pattern);

const Vec3b RED		(  0.0,   0.0, 255.0);
const Vec3b GREEN	(  0.0, 255.0,   0.0);
const Vec3b BLUE	(255.0,   0.0,   0.0);
const Vec3b YELLOW	(  0.0, 255.0, 255.0);
const Vec3b MAGENTA	(255.0,   0.0, 255.0);
const Vec3b CYAN	(255.0, 255.0,   0.0);
const Vec3b WHITE	(255.0, 255.0, 255.0);
const Vec3b GRAY	(150.0, 150.0, 150.0);
const Vec3b BLACK	(  0.0,   0.0,   0.0);



// Opens the image and handles potential error
Mat_<uchar> openGrayscaleImage() {
	Mat_<uchar> img = imread(IMAGE_PATH, IMREAD_GRAYSCALE);
	
	if (img.rows == 0 || img.cols == 0) {
		printf("Could not open image");
		exit(1);
	}

	if (SHOW_OPEN_GRAYSCALE_IMAGE) {
		imshow("Open Grayscale Image", img);
	}

	return img;
}


// Check if pixel at location (i,j) is inside the picture
bool isInside(Mat img, int i, int j) {
	return (i >= 0 && i < img.rows) && (j >= 0 && j < img.cols);
}


// Convert grayscale image to binary based on threshold
Mat_<uchar> convertToBinary(Mat_<uchar> img) {
	Mat_<uchar> imgRes(img.rows, img.cols);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			if (img(i, j) < THRESHOLD_FOR_BINARY) {
				imgRes(i, j) = 0;	// black
			}
			else {
				imgRes(i, j) = 255;	// white
			}
		}
	}

	if (SHOW_CONVERT_TO_BINARY) {
		imshow("Convert To Binary", imgRes);
	}

	return imgRes;
}


// Return the horizontal projection: hp[i] = number of pixels on row i
std::vector<int> getHorizontalProjection(Mat_<uchar> img) {
	Mat_<uchar> imgRes(img.rows, img.cols);
	std::vector<int> horizontalProjection(img.rows);

	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			// if object pixel
			if (img(i, j) == 0) {
				//increment the value of horizontalProjection at index i and store in an image to visualize
				imgRes(i, horizontalProjection[i]++) = 0;	
			}
		}
	}

	if (SHOW_HORIZONTAL_PROJECTION) {
		imshow("Horizontal Projection", imgRes);
	}

	return horizontalProjection;
}


// Extract the music sheet lines (vector of y coordinates)
std::vector<int> getLines(Mat_<uchar> img, std::vector<int> horizontalProjection) {
	std::vector<int> lineCoordinates;
	
	int threshold = img.cols * THRESHOLD_FOR_LINE;

	for (int i = 0; i < horizontalProjection.size(); i++) {
		if (horizontalProjection[i] > threshold) {
			lineCoordinates.push_back(i);
			
			// skip redundant lines
			while (i < horizontalProjection.size() && horizontalProjection[i] > threshold) {
				i++;
			}
		}
	}


	if (SHOW_EXTRACT_LINES) {
		Mat_<Vec3b> imgRes(img.rows, img.cols);

		// copy the binary image but with gray
		for (int i = 0; i < img.rows; i++) {
			for (int j = 0; j < img.cols; j++) {
				if (img(i, j) == 0) {
					imgRes(i, j) = GRAY;
				}
			}
		}

		int ci = 0;							// color index
		Vec3b colors[] = { RED, GREEN };	// colors used for "classes" of lines (5 by 5)
		Vec3b color = colors[ci++];			// extract first color, increment ci

		for (int i = 0; i < lineCoordinates.size(); i++) {
			// draw a music sheet line
			line(imgRes, Point(0, lineCoordinates[i]), Point(img.cols, lineCoordinates[i]), color);
			
			// change color if 5 lines were drawn
			if ((i + 1) % 5 == 0) {
				color = colors[(ci++) % 2];
			}
		}

		imshow("Extract Lines", imgRes);
	}

	return lineCoordinates;
}


// Perform a dilation on img, with structuring element sel
Mat_<uchar> dilation(Mat_<uchar> img, Mat_<uchar> sel) {
	Mat_<uchar> imgRes(img.rows, img.cols, 255);

	// iterate original pixels
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			// only consider object pixels from original
			if (img(i, j) != 0) {
				continue;
			}

			// iterate pixels given by structuring element
			//		"If the origin of the structuring element coincides with an object pixel in the image, 
			//		 label all pixels covered by the structuring element as object pixels in the result image."
			for (int u = 0; u < sel.rows; u++) {
				for (int v = 0; v < sel.cols; v++) {
					// only consider object pixels from structuring element
					if (sel(u, v) != 0) {
						continue;
					}

					// offset structuring element's pixel
					int i2 = u - sel.rows / 2 + i;
					int j2 = v - sel.cols / 2 + j;

					if (isInside(imgRes, i2, j2)) {
						imgRes(i2, j2) = 0;
					}
				}
			}
		}
	}

	if (SHOW_DILATION) {
		imshow("Dilation", imgRes);
	}

	return imgRes;
}


// Perform an erosion on img, with structuring element sel
Mat_<uchar> erosion(Mat_<uchar> img, Mat_<uchar> sel) {
	Mat_<uchar> imgRes(img.rows, img.cols, 255);

	// iterate original pixels
	for (int i = 0; i < img.rows; i++) {
		for (int j = 0; j < img.cols; j++) {
			// only consider object pixels from original
			if (img(i, j) != 0) {
				continue;
			}

			// iterate pixels given by structuring element
			//		"If the structuring element covers any background pixel,
			//		 the pixel in the result image keeps its background label."
			for (int u = 0; u < sel.rows; u++) {
				for (int v = 0; v < sel.cols; v++) {
					// only consider object pixels from structuring element
					if (sel(u, v) != 0) {
						continue;
					}

					// offset structuring element's pixel
					int i2 = u - sel.rows / 2 + i;
					int j2 = v - sel.cols / 2 + j;

					if (isInside(imgRes, i2, j2) && img(i2, j2) != 0) {
						goto skip;
					}
				}
			}
			imgRes(i, j) = 0;
		skip:;
		}
	}

	if (SHOW_EROSION) {
		imshow("Erosion", imgRes);
	}

	return imgRes;
}

// "Opening consists of an erosion followed by a dilation and can be used to eliminate 
//  all pixels in regions that are too small to contain the structuring element."
Mat_<uchar> opening(Mat_<uchar> img, Mat_<uchar> sel) {
	Mat_<uchar> imgAux = erosion(img, sel);
	Mat_<uchar> imgRes = dilation(imgAux, sel);

	if (SHOW_OPENING) {
		imshow("Opening", imgRes);
	}

	return imgRes;
}



int main() {
	Mat_<uchar> originalImage = openGrayscaleImage();
	Mat_<uchar> binaryImg = convertToBinary(originalImage);
	std::vector<int> horizontalProjection = getHorizontalProjection(binaryImg);
	std::vector<int> lines = getLines(binaryImg, horizontalProjection);
	Mat_<uchar> openingImg = opening(binaryImg, sel);

	waitKey(0);
	return 0;
}