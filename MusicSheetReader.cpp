#include <opencv2/opencv.hpp>
#include <random>		// needed for pseudorandom label coloring
#include <iostream>		// for writing to standard output
#include <fstream>		// for writing to file
#include <string>
#include <algorithm>	// for sort

#define RUN_PYTHON_SCRIPT true
#define PYTHON_COMMAND "python3 /home/broland/Documents/ut/ip/music_sheet_reader_py/NotesToMidi.py"

#define IMAGE_PATH "Images/mary.bmp"		// path of image being processed
#define THRESHOLD_FOR_BINARY 150				// below this, consider black (0, object) pixel, above white (1, background) pixel
#define THRESHOLD_FOR_LINE 0.5					// a row is considered to have a music sheet line if it has more than this percent (of the image width) pixels black

#define SHOW_GRAYSCALE_IMAGE false
#define SHOW_BINARY_IMAGE true
#define SHOW_HORIZONTAL_PROJECTION false
#define SHOW_LINES false
#define SHOW_DILATION false
#define SHOW_EROSION false
#define SHOW_OPENING false
#define SHOW_CONNECTED_COMPONENTS_BFS true
#define SHOW_AREA false
#define SHOW_CENTER_OF_MASS false
#define SHOW_ALL_NOTES true

#define MIN_NOTE_AREA 25
#define MAX_NOTE_AREA 45
#define LINE_OFFSET_TOLERANCE 1


// structuring element for opening
uchar pattern[25] = {
        255,	255,		0,		255,	255,
        255,	  0,		0,		  0,	255,
          0,	  0,		0,		  0,	  0,
        255,	  0,		0,		  0,	255,
        255,	255,		0,		255,	255,
};
const cv::Mat_<uchar> sel = cv::Mat(5, 5, CV_8UC1, pattern);


// duration of a musical note_
enum duration_ { whole, half, quarter, eighth ,sixteenth };

// actual "value", "name" of musical note_
enum name_ { C, D, E, F, G, A, B };


// structure for a musical note_
struct note_ {
    name_ name;
    int octave;			// most usual is 4th
    duration_ duration;
};

// structure for an extracted line
struct line_ {
    int y;				// y coordinate of the line on the image
};

// lines grouped by 5 (a staff_), upmost staff_ is 0
// lines defined by index, uppermost is 0 (E4 as note_), lowermost is 4 (A5 as note_)
struct staff_ {
    line_ lines[5];
};



std::string encodeNote(note_ n) {
    char encoding[4];
    switch (n.name) {
        case C: encoding[0] = 'C'; break;
        case D: encoding[0] = 'D'; break;
        case E: encoding[0] = 'E'; break;
        case F: encoding[0] = 'F'; break;
        case G: encoding[0] = 'G'; break;
        case A: encoding[0] = 'A'; break;
        case B: encoding[0] = 'B'; break;
    }
    switch (n.octave) {
        case 4: encoding[1] = '4'; break;
        case 5: encoding[1] = '5'; break;
    }
    switch (n.duration) {
        case whole:     encoding[2] = 'W'; break;
        case half:      encoding[2] = 'H'; break;
        case quarter:   encoding[2] = 'Q'; break;
        case eighth:    encoding[2] = 'E'; break;
        case sixteenth: encoding[2] = 'S'; break;
    }
    encoding[3] = '\0';
    return encoding;  // cast from char array to string implicit
}


// Check if pixel at location (i,j) is inside the picture
bool isInside(cv::Mat img, int i, int j) {
    return (i >= 0 && i < img.rows) && (j >= 0 && j < img.cols);
}


// Open the image and handles potential error
cv::Mat_<uchar> openGrayscaleImage() {
    cv::Mat_<uchar> img = cv::imread(IMAGE_PATH,cv::IMREAD_GRAYSCALE);

    if (img.rows == 0 || img.cols == 0) {
        printf("Could not open image\n");
        exit(1);
    }

    if (SHOW_GRAYSCALE_IMAGE) {
        imshow("Open Grayscale Image", img);
    }

    return img;
}


// Convert grayscale image to binary based on threshold
cv::Mat_<uchar> convertToBinary(cv::Mat_<uchar> img) {
    cv::Mat_<uchar> imgRes(img.rows, img.cols);

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

    if (SHOW_BINARY_IMAGE) {
        imshow("Convert To Binary", imgRes);
    }

    return imgRes;
}


// Return a binary image with black values 230 instead of 0 (visualization purposes)
cv::Mat_<uchar> copyImageWithGrayUchar(cv::Mat_<uchar> img) {
    cv::Mat_<uchar> imgRes(img.rows, img.cols);

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (img(i, j) == 0) {
                imgRes(i, j) = 230;
            }
            else {
                imgRes(i, j) = 255;
            }
        }
    }

    return imgRes;
}


// Return a binary image with black values 230 instead of 0 (visualization purposes)
cv::Mat_<cv::Vec3b> copyImageWithGrayVec3b(cv::Mat_<uchar> img) {
    cv::Mat_<cv::Vec3b> imgRes(img.rows, img.cols);

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (img(i, j) == 0) {
                imgRes(i, j) = cv::Vec3b(230.0, 230.0, 230.0);
            }
            else {
                imgRes(i, j) = cv::Vec3b(255.0, 255.0, 255.0);
            }
        }
    }

    return imgRes;
}


// "Put imgTop on imgBottom", return result
cv::Mat_<uchar> overlayImages(cv::Mat_<uchar> imgBottom, cv::Mat_<uchar> imgTop) {
    cv::Mat_<uchar> imgRes(imgBottom.rows, imgBottom.cols);

    for (int i = 0; i < imgBottom.rows; i++) {
        for (int j = 0; j < imgBottom.cols; j++) {
            if (imgTop(i, j) != 255) {  // imgTop has priority over resulting value, except if background
                imgRes(i, j) = imgTop(i, j);
            }
            else {
                imgRes(i, j) = imgBottom(i, j);
            }
        }
    }

    return imgRes;
}


// Return the horizontal projection: hp[i] = number of pixels on row i
std::vector<int> getHorizontalProjection(cv::Mat_<uchar> img) {
    std::vector<int> horizontalProjection(img.rows);

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (img(i, j) == 0) {
                horizontalProjection[i]++;
            }
        }
    }

    if (SHOW_HORIZONTAL_PROJECTION) {
        cv::Mat_<uchar> imgRes = copyImageWithGrayUchar(img);
        for (int i = 0; i < horizontalProjection.size(); i++) {
            for (int j = 0; j < horizontalProjection[i]; j++) {
                imgRes(i, j) = 0;
            }
        }
        cv::imshow("Horizontal Projection", imgRes);
    }

    return horizontalProjection;
}

std::vector<int> linesOverThreshold;

// Extract the music sheet lines
std::vector<staff_> getStaffs(cv::Mat_<uchar> img, std::vector<int> horizontalProjection) {
    std::vector<staff_> staffs;

    int threshold = img.cols * THRESHOLD_FOR_LINE;
    int lineCounter = 0;
    staff_ currentStaff = {};

    for (int i = 0; i < horizontalProjection.size(); i++) {
        // if we meet a line
        if (horizontalProjection[i] > threshold) {
            currentStaff.lines[lineCounter % 5] = line_ { i };
            linesOverThreshold.push_back(i);

            // skip "redundant" lines
            i++;
            while (i < horizontalProjection.size() && horizontalProjection[i] > threshold) {
                linesOverThreshold.push_back(i);
                i++;
            }

            lineCounter++;

            // if all 5 lines found, save currentStaff and reset it
            if (lineCounter % 5 == 0) {
                staffs.push_back(currentStaff);
                currentStaff = {};
            }
        }
    }

    if (SHOW_LINES) {
        cv::Mat_<cv::Vec3b> imgRes = copyImageWithGrayVec3b(img);

        // use two colors to somewhat distinguish nearby staffs
        cv::Vec3b colors[] = {
                cv::Vec3b(255.0,   0.0,   0.0), // blue
                cv::Vec3b(  0.0,   0.0, 255.0)  // red
        };

        for (int staffNo = 0; staffNo < staffs.size(); staffNo++) {
            for (line_ line : staffs[staffNo].lines) {
                // draw a music sheet line
                cv::line(
                        imgRes, cv::Point(0, line.y),
                        cv::Point(img.cols, line.y),
                        colors[staffNo % 2]
                );
            }
        }
        imshow("Extract Staffs", imgRes);
    }

    return staffs;
}


// Perform erosion on img, with structuring element sel
cv::Mat_<uchar> erosion(cv::Mat_<uchar> img, cv::Mat_<uchar> sel) {
    cv::Mat_<uchar> erosionImg(img.rows, img.cols, 255);

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

                    if (isInside(erosionImg, i2, j2) && img(i2, j2) != 0) {
                        goto skip;
                    }
                }
            }
            erosionImg(i, j) = 0;
            skip:;
        }
    }

    if (SHOW_EROSION) {
        cv::Mat_<uchar> resImg = copyImageWithGrayUchar(img);
        resImg = overlayImages(resImg, erosionImg);
        cv::imshow("Erosion", resImg);
    }

    return erosionImg;
}


// Perform dilation on img, with structuring element sel
cv::Mat_<uchar> dilation(cv::Mat_<uchar> img, cv::Mat_<uchar> sel) {
    cv::Mat_<uchar> dilationImg(img.rows, img.cols, 255);

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

                    if (isInside(dilationImg, i2, j2)) {
                        dilationImg(i2, j2) = 0;
                    }
                }
            }
        }
    }

    if (SHOW_DILATION) {
        // effect will not be seen as dilationImg fully covers img (eroded image)
        cv::Mat_<uchar> resImg = copyImageWithGrayUchar(img);
        resImg = overlayImages(resImg, dilationImg);
        cv::imshow("Dilation", resImg);
    }

    return dilationImg;
}


// "Opening consists of an erosion followed by a dilation and can be used to eliminate
//  all pixels in regions that are too small to contain the structuring element."
cv::Mat_<uchar> opening(cv::Mat_<uchar> img, cv::Mat_<uchar> sel) {
    cv::Mat_<uchar> imgAux = erosion(img, sel);
    cv::Mat_<uchar> imgRes = dilation(imgAux, sel);

    if (SHOW_OPENING) {
        imshow("Opening", imgRes);
    }

    return imgRes;
}


// Search for connected components in img using Breadth First Traversal
// Note that the direction of going through the image corresponds to the direction of reading a music sheet
cv::Mat_<int> connectedComponentsBFS(cv::Mat_<uchar> img, std::vector<staff_> staffs, int &maxLabel) {
    int currentLabel = 0;						// counter for labeling
    cv::Mat_<int> labelsImg(img.rows, img.cols, 0);	// labels of corresponding pixels, initially all 0s (unlabeled)

    // index offsets for 8-neighborhood
    int di[8] = { -1, -1, -1, 0, 1, 1,  1,  0 };
    int dj[8] = { -1,  0,  1, 1, 1, 0, -1, -1 };

    int pi, pj;	// pixel index
    int ni, nj;	// neighbor index

    std::vector<std::pair<int, int>> staffRanges;
    for (staff_ s : staffs) {
        int upperBound = s.lines[0].y - LINE_OFFSET_TOLERANCE;
        int lowerBound = s.lines[4].y + LINE_OFFSET_TOLERANCE;
        if (upperBound < 0) {
            upperBound = 0;
        }
        if (lowerBound > img.rows) {
            lowerBound = img.rows;
        }

        for (int j = 0; j < img.cols; j++) {
            for (int i = upperBound; i <= lowerBound; i++) {
                // discard non-object and already labeled pixels
                if (img(i, j) != 0 || labelsImg(i, j) != 0) {
                    continue;
                }

                // start of a new connected component (new BFS)
                std::queue<std::pair<int, int>> Q;

                // label pixel and enqueue
                labelsImg(i, j) = ++currentLabel;
                Q.push(std::pair<int, int>(i, j));

                while (!Q.empty()) {
                    // dequeue and decompose
                    std::pair<int, int> p = Q.front();
                    pi = p.first;
                    pj = p.second;
                    Q.pop();

                    // for each neighbor
                    for (int k = 0; k < 8; k++) {
                        ni = pi + di[k];
                        nj = pj + dj[k];

                        // discard out of bounds neighbors
                        if (!isInside(img, ni, nj)) {
                            continue;
                        }

                        // discard non-object and already labeled neighbor pixels
                        if (img(ni, nj) != 0 || labelsImg(ni, nj) != 0) {
                            continue;
                        }

                        labelsImg(ni, nj) = currentLabel;
                        Q.push(std::pair<int, int>(ni, nj));
                    }
                }
            }
        }
    }

    if (SHOW_CONNECTED_COMPONENTS_BFS) {
        // generate random colors
        std::default_random_engine gen;
        std::uniform_int_distribution<int> d(0, 255);
        std::vector<cv::Vec3b> colors(currentLabel + 1);	// labeling has range [0, currentLabel]

        colors[0] = cv::Vec3b(255.0, 255.0, 255.0);  // we consider 0 unlabeled, i.e. background
        bool seenLabel[currentLabel + 1];
        for (int i = 1; i <= currentLabel; i++) {
            colors[i] = cv::Vec3b(d(gen), d(gen), d(gen));	// other labels have random color
            seenLabel[i] = false;
        }

        //int count = 0;
        cv::Mat_<cv::Vec3b> colorImg(labelsImg.rows, labelsImg.cols);
        for (int i = 0; i < labelsImg.rows; i++) {
            for (int j = 0; j < labelsImg.cols; j++) {
                int label = labelsImg(i, j);
                cv::Vec3b color = colors[label];

                colorImg(i, j) = colors[label];

                if (!seenLabel[label]) {
                    cv::putText(
                            colorImg,
                            std::to_string(label),
                            cv::Point(j, i),
                            cv::FONT_HERSHEY_COMPLEX,
                            0.5,
                            cv::Scalar(color[0], color[1], color[2]),
                            1,
                            false);
                    seenLabel[label] = true;
                }
            }
        }



        cv::imshow("Connected Components BFS", colorImg);
    }

    maxLabel = currentLabel;
    return labelsImg;
}


// Compute the area of a binary object
int area(cv::Mat_<uchar> img) {
    int area = 0;

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (img(i, j) == 0) {
                area++;
            }
        }
    }

    return area;
}


// Compute the center of mass of binary object
cv::Point2i centerOfMass(cv::Mat_<uchar> img) {
    cv::Point2i com(0, 0);

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (img(i, j) == 0) {
                com.x += j;  // x coordinate corresponds to column
                com.y += i;  // y coordinate corresponds to row
            }
        }
    }

    float A = area(img);
    com.x /= A;
    com.y /= A;

    return com;
}


// draw a cross on image img, "around" point p, with given diameter
void drawCross(cv::Mat_<uchar> img, cv::Point2i p, int diameter, int color=255) {
    // calculate potential end coordinates of cross
    int halfDiameter = diameter / 2;
    int xl = p.x - halfDiameter;	// x left
    int xr = p.x + halfDiameter;	// x right
    int yt = p.y - halfDiameter;	// y top
    int yb = p.y + halfDiameter;	// y bottom

    // make sure end points inside img
    if (xl < 0) {
        xl = 0;
    }
    if (xr > img.cols) {
        xr = img.cols;
    }
    if (yt < 0) {
        yt = 0;
    }
    if (yb > img.rows) {
        yb = img.rows;
    }

    // define points for drawing line and draw lines
    cv::Point2i l(xl, p.y);
    cv::Point2i r(xr, p.y);
    cv::Point2i t(p.x, yt);
    cv::Point2i b(p.x, yb);
    cv::line(img, l, r, color);
    cv::line(img, t, b, color);
}


cv::Mat_<uchar> extractComponent(cv::Mat_<int> labelImg, int label) {
    cv::Mat_<uchar> resImg(labelImg.rows, labelImg.cols);

    for (int i = 0; i < labelImg.rows; i++) {
        for (int j = 0; j < labelImg.cols; j++) {
            if (labelImg(i, j) == label) {
                resImg(i, j) = 0;
            }
            else {
                resImg(i, j) = 255;
            }
        }
    }

    return resImg;
}


duration_ getDuration(cv::Mat_<uchar> img, cv::Point2i com, cv::Mat_<uchar> flagImg) {

    /* ez jo
    uchar pattern1[25] = {
            255,  255,		0,		255,	255,
            255,  255,		0,		255,	255,
            255,  255,		0,		255,	255,
            255,  255,		0,		255,	255,
            255,  255,		0,		255,	255,
    };
    const cv::Mat_<uchar> sel1 = cv::Mat(5, 5, CV_8UC1, pattern1);
    */
    uchar pattern1[12] = {
              255,		0,		255,
              255,		0,		255,
              255,		0,		255,
              255,		0,		255,
    };
    const cv::Mat_<uchar> sel1 = cv::Mat(4, 3, CV_8UC1, pattern1);

    cv::Mat_<uchar> noLinesImg = opening(img, sel1);

    cv::imshow("noline", noLinesImg);


    uchar pattern2[12] = {
            255,  255,		255,
            0,  0,		0,
            0,  0,		0,
            255,  255,		255,
    };

    uchar pattern3[9] = {
            255,  0,		255,
            0, 0,		0,
            255, 0, 255
    };


    const cv::Mat_<uchar> sel2 = cv::Mat(4, 3, CV_8UC1, pattern2);
    const cv::Mat_<uchar> sel3 = cv::Mat(3, 3, CV_8UC1, pattern3);

    //cv::Mat_<uchar> aux = dilation(img, sel2);
    //cv::Mat_<uchar> aux2 = dilation(aux, sel2);
    //cv::Mat_<uchar> aux = dilation(img, sel3);

    cv::Mat_<uchar> aux = erosion(img, sel2);
    cv::Mat_<uchar> noLinesImg2 = dilation(aux, sel2);


    //cv::Mat_<uchar> noLinesImg2 = erosion(img, sel2);


    //cv::Mat_<uchar> noLinesImg2 = erosion(aux, sel2);

    //cv::Mat_<uchar> noLinesImg2 = opening(img, sel2);
    cv::imshow("noline2", noLinesImg2);

    // start of a new connected component (new BFS)
    std::queue<cv::Point2i> Q;
    Q.push(com);

    // index offsets for 8-neighborhood
    int di[8] = { -1, -1, -1, 0, 1, 1,  1,  0 };
    int dj[8] = { -1,  0,  1, 1, 1, 0, -1, -1 };

    int pi, pj;	// pixel index
    int ni, nj;	// neighbor index

    cv::Mat_<uchar> compImg(img.rows, img.cols);
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            compImg(i, j) = 255;
        }
    }



    //int NOTE_NEIGHBORHOOD = 1000;
    while (!Q.empty()) {
        //printf("value at COM: %d\n", img());
        // dequeue and decompose
        cv::Point2i p = Q.front();
        pi = p.y;
        pj = p.x;
        Q.pop();

        // for each neighbor
        for (int k = 0; k < 8; k++) {
            ni = pi + di[k];
            nj = pj + dj[k];

            //if (nj < com.x - NOTE_NEIGHBORHOOD || nj > com.x + NOTE_NEIGHBORHOOD) {
            //    continue;
            //}

            // discard out of bounds neighbors
            if (!isInside(img, ni, nj)) {
                continue;
            }

            // discard non-object and already labeled neighbor pixels
            if (noLinesImg(ni, nj) != 0 || compImg(ni, nj) == 0) {
                continue;
            }

            compImg(ni, nj) = 0;
            Q.push(cv::Point2i (nj, ni));
        }
    }

    cv::imshow("labelsimg", compImg);
    cv::Point2i newCom = centerOfMass(compImg);

    int endX, endY;
    if (newCom.y < com.y) {
        // looking for uppermost point -> first one we meet
        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                if (compImg(i, j) == 0) {
                    endX = j;
                    endY = i;
                    goto found;
                }
            }
        }
    }
    else {
        // looking for lowermost point -> last one we meet
        for (int i = 0; i < img.rows; i++) {
            for (int j = 0; j < img.cols; j++) {
                if (compImg(i, j) == 0) {
                    endX = j;
                    endY = i;
                }
            }
        }
    }
    found:
    cv::Point2i endPoint = cv::Point2i(endX, endY);

    //drawCross(flagImg, newCom, 10, 0);

    //drawCross(imgRes, lastPoint, 10)

    int xOffset = 3;
    int yOffset = 1;
    int fx, fy;

    // may have stem under note head
    if (newCom.y > com.y) {
        fy = endPoint.y - yOffset;
        fx = endPoint.x + xOffset;  // check to the right

        if(std::find(linesOverThreshold.begin(), linesOverThreshold.end(), fy) != linesOverThreshold.end()) {
            return quarter;
        }

        // todo have staff here, check if y line, if so, skip until not line (increment/decrement)
        if (img(fy, fx) == 0) {
            drawCross(flagImg, cv::Point2i(fx, fy), 10, 50);
            return eighth;
        }

        fx = endPoint.x - xOffset;  // check to the left
        if (img(fy, fx) == 0) {
            drawCross(flagImg, cv::Point2i(fx, fy), 10, 50);
            return eighth;
        }

        //drawCross(flagImg, cv::Point2i(fx, fy), 10, 50);
        return quarter;
    }

    // may have stem over note head
    fy = endPoint.y + yOffset;
    fx = endPoint.x + xOffset;  // check to the right
    if(std::find(linesOverThreshold.begin(), linesOverThreshold.end(), fy) != linesOverThreshold.end()) {
        return quarter;
    }

    if (img(fy, fx) == 0) {
        drawCross(flagImg, cv::Point2i(fx, fy), 10, 50);
        return eighth;
    }

    fx = endPoint.x - xOffset;  // check to the left
    if (img(fy, fx) == 0) {
        drawCross(flagImg, cv::Point2i(fx, fy), 10, 50);
        return eighth;
    }

    //drawCross(flagImg, cv::Point2i(fx, fy), 10, 50);
    return quarter;



    //cv::imshow("Note", imgRes);

}


std::vector<note_> extractNotes(cv::Mat_<int> binaryImg, cv::Mat_<int> labelImg, int maxLabel, std::vector<staff_> staffs) {
    cv::Mat_<uchar> crossImg = cv::Mat::zeros(labelImg.rows, labelImg.cols, CV_8UC1);

    std::vector<int> noteLabels;
    std::vector<note_> notes;

    cv::Mat_<uchar> flagImg = copyImageWithGrayUchar(binaryImg);

    // go through labels, skip 0 (background)
    for (int label = 1; label <= maxLabel; label++) {
        cv::Mat_<uchar> componentImg = extractComponent(labelImg, label);

        // check area criterion
        int a = area(componentImg);
        if (a > MAX_NOTE_AREA || a < MIN_NOTE_AREA) {
            continue;
        }
        if (SHOW_AREA) {
            std::cout << a << std::endl;
        }

        // check center of mass criterion
        cv::Point2i com = centerOfMass(componentImg);
        if (com.y < staffs[0].lines[0].y) {
            continue;
        }
        if (com.x < 62) {
            continue;
        }
        if (SHOW_CENTER_OF_MASS) {
            drawCross(crossImg, com, 50);
        }

        //if (label == 10) {
        duration_ duration = getDuration(binaryImg, com, flagImg);
        //}

        // passed all requirements => is considered a note_
        // printf("%d: (%d,%d)\n", label, com.x, com.y);

        int tolerance = 1;
        int maxOffset = 5;
        name_ n;
        int octave;

        bool processed = false;
        int staffNo = 0;
        while (staffNo < staffs.size() && !processed) {
            staff_ s = staffs[staffNo++];

            if (com.y < s.lines[0].y - maxOffset || com.y > s.lines[4].y + maxOffset) {
                // out of current staff_'s range, continue searching in next staff_
                continue;
            }

            // inside current staff_'s range, associate name and octave, and quit searching
            if (com.y < s.lines[0].y - tolerance) {
                n = G;
                octave = 5;
                processed = true;
            }
            else if (com.y < s.lines[0].y + tolerance) {
                n = F;
                octave = 5;
                processed = true;
            }
            else if (com.y < s.lines[1].y - tolerance) {
                n = E;
                octave = 5;
                processed = true;
            }
            else if (com.y < s.lines[1].y + tolerance) {
                n = D;
                octave = 5;
                processed = true;
            }
            else if (com.y < s.lines[2].y - tolerance) {
                n = C;
                octave = 5;
                processed = true;
            }
            else if (com.y < s.lines[2].y + tolerance) {
                n = B;
                octave = 4;
                processed = true;
            }
            else if (com.y < s.lines[3].y - tolerance) {
                n = A;
                octave = 4;
                processed = true;
            }
            else if (com.y < s.lines[3].y + tolerance) {
                n = G;
                octave = 4;
                processed = true;
            }
            else if (com.y < s.lines[4].y - tolerance) {
                n = F;
                octave = 4;
                processed = true;
            }
            else if (com.y < s.lines[4].y + tolerance) {
                n = E;
                octave = 4;
                processed = true;
            }
            else {
                n = D;
                octave = 4;
                processed = true;
            }
        }

        if (!processed) {
            std::cout << "Could not process point with y " << com.y << "." << std::endl;
            continue;
        }

        noteLabels.push_back(label);
        notes.push_back(note_{ n, octave, duration });
    }

    cv::imshow("Flags", flagImg);

    if (SHOW_CENTER_OF_MASS) {
        imshow("CenterOfMass", crossImg);
    }

    if (SHOW_ALL_NOTES) {
        cv::Mat_<uchar> noteImg(labelImg.rows, labelImg.cols);
        for (int i = 0; i < labelImg.rows; i++) {
            for (int j = 0; j < labelImg.cols; j++) {
                // https://stackoverflow.com/questions/3450860/check-if-a-stdvector-contains-a-certain-object
                if (std::find(noteLabels.begin(), noteLabels.end(), labelImg(i, j)) != noteLabels.end()) {
                    noteImg(i, j) = 0;
                }
                else {
                    noteImg(i, j) = 255;
                }
            }
        }
        cv::Mat_<uchar> resImg = copyImageWithGrayUchar(binaryImg);
        resImg = overlayImages(resImg, noteImg);
        imshow("All Notes", resImg);
    }

    return notes;
}

void writeNotesToFile(std::vector<note_> notes) {
    std::ofstream outFile;
    outFile.open("notes.txt");

    if (SHOW_ALL_NOTES) {
        std::cout << "Encoded notes:" << std::endl;
    }

    for (note_ n : notes) {
        std::string encodedNote = encodeNote(n);
        if (SHOW_ALL_NOTES) {
            std::cout << encodedNote << std::endl;
        }
        outFile << encodeNote(n) << std::endl;
    }
    outFile.close();
}

int main() {
    cv::Mat_<uchar> originalImage = openGrayscaleImage();
    cv::Mat_<uchar> binaryImg = convertToBinary(originalImage);
    std::vector<int> horizontalProjection = getHorizontalProjection(binaryImg);
    std::vector<staff_> staffs = getStaffs(binaryImg, horizontalProjection);
    cv::Mat_<uchar> openingImg = opening(binaryImg, sel);
    int maxLabel;
    cv::Mat_<int> labelImg = connectedComponentsBFS(openingImg, staffs, maxLabel);
    //printf("%d", maxLabel);
    std::vector<note_> notes = extractNotes(binaryImg, labelImg, maxLabel, staffs);
    writeNotesToFile(notes);

    /*
    note_ n1 = makeNote(0, cv::Point2i(), "D4Q");
    note_ n2 = makeNote(0, cv::Point2i(), "G4E");
    note_ n3 = makeNote(0, cv::Point2i(), "G4E");
    note_ n4 = makeNote(0, cv::Point2i(), "G4Q");
    note_ n5 = makeNote(0, cv::Point2i(), "A4Q");
    std::vector<note_> notes{n1, n2, n3, n4, n5};
    writeNotesToFile(notes);
    */

    //waitKey(0);

    if (RUN_PYTHON_SCRIPT) {
        system(PYTHON_COMMAND);
    }

    cv::waitKey(0);
    return 0;
}
