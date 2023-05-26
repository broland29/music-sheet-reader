#include <opencv2/opencv.hpp>       // include opencv on linux
#include <random>		            // for random colors for connected components
#include <iostream>		            // for printing to standard output
#include <fstream>		            // for writing notes.txt
#include <string>                   // for easier handling of encoded notes


#define RUN_PYTHON_SCRIPT true
#define PYTHON_COMMAND "python3 /home/broland/Documents/ut/ip/music_sheet_reader_py/NotesToMidi.py"

#define IMAGE_PATH "Images/tannenbaum.bmp"		// path of image being processed
#define THRESHOLD_FOR_BINARY 150				// below object pixel, above background pixel
#define THRESHOLD_FOR_LINE 0.5					// lines with image_width * this value are considered lines

#define MIN_NOTE_AREA 25                        // in order to be considered a note head, must have
#define MAX_NOTE_AREA 45                        //  MIN_NOTE_AREA < area < MAX_NOTE_AREA
#define LINE_OFFSET_TOLERANCE 1                 // connected components with greater offset from a staff are discarded
#define MIN_X_NOTE_HEAD 62                      // everything on the left side of this is discarded

#define SHOW_GRAYSCALE_IMAGE false
#define SHOW_BINARY_IMAGE true
#define SHOW_HORIZONTAL_PROJECTION false
#define SHOW_STAFFS true
#define SHOW_OPENING false
#define SHOW_CONNECTED_COMPONENTS_BFS true
#define SHOW_AREA false
#define SHOW_NO_LINE true
#define SHOW_CENTER_OF_MASS false
#define SHOW_FLAGS false
#define SHOW_ALL_NOTES true
#define SHOW_NOTE_ENCODINGS false


uchar noteHeadPattern[25] = {
        255,	255,		0,		255,	255,
        255,	  0,		0,		  0,	255,
        0,	      0,		0,		  0,	  0,
        255,	  0,		0,		  0,	255,
        255,	255,		0,		255,	255,
};
const cv::Mat_<uchar> noteHeadStructuringElement = cv::Mat(5, 5, CV_8UC1, noteHeadPattern);

uchar stemPattern[12] = {
        255,		0,		255,
        255,		0,		255,
        255,		0,		255,
        255,		0,		255,
};
const cv::Mat_<uchar> stemStructuringElement = cv::Mat(4, 3, CV_8UC1, stemPattern);


// duration of a musical note
enum duration_ { whole, half, quarter, eighth, sixteenth };

// actual "value", "name" of musical note
enum name_ { C, D, E, F, G, A, B };

// structure for a musical note
struct note_ {
    name_ name;
    int octave;	            // most common is 4th
    duration_ duration;     // most common is quarter
};

// structure for an extracted line
struct line_ {
    int y;				    // the y coordinate of the line on the image
};

// lines grouped by 5 (a staff), upmost staff is 0
// lines defined by index, uppermost is 0 (E4 as note), lowermost is 4 (A5 as note)
struct staff_ {
    line_ lines[5];
};


// Given a note n as input return its encoding for passing on to the python script
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
bool isInside(const cv::Mat& img, int i, int j) {
    return (i >= 0 && i < img.rows) && (j >= 0 && j < img.cols);
}


// Open the image and handle potential error
cv::Mat_<uchar> openGrayscaleImage() {
    cv::Mat_<uchar> img = cv::imread(IMAGE_PATH,cv::IMREAD_GRAYSCALE);

    if (img.rows == 0 || img.cols == 0) {
        printf("Could not open image\n");
        exit(1);
    }

    if (SHOW_GRAYSCALE_IMAGE) {
        imshow("Grayscale Image", img);
    }

    return img;
}


// Convert grayscale image to binary based on threshold
cv::Mat_<uchar> convertToBinary(cv::Mat_<uchar> img) {
    cv::Mat_<uchar> imgRes(img.rows, img.cols);

    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            if (img(i, j) < THRESHOLD_FOR_BINARY) {
                imgRes(i, j) = 0;
            }
            else {
                imgRes(i, j) = 255;
            }
        }
    }

    if (SHOW_BINARY_IMAGE) {
        imshow("Binary Image", imgRes);
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


// Same as copyImageWithGrayUchar but for three channel color images
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


// "Put imgTop on imgBottom", return result (for visualization purposes)
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


// Return the horizontal projection: horizontalProjection[i] = number of pixels on row i
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


// Get a vector of all lines which satisfy the threshold
std::vector<int> getLinesOverThreshold(const cv::Mat_<uchar>& img, std::vector<int> horizontalProjection) {
    std::vector<int> linesOverThreshold;
    int threshold = img.cols * THRESHOLD_FOR_LINE;

    for (int i = 0; i < horizontalProjection.size(); i++) {
        if (horizontalProjection[i] > threshold) {
            linesOverThreshold.push_back(i);
        }
    }

    return linesOverThreshold;
}


// Process the possible lines: extract actual lines and group them in staffs
std::vector<staff_> getStaffs(const cv::Mat_<uchar>& img, std::vector<int> linesOverThreshold) {
    std::vector<staff_> staffs;

    int lineCounter = 0;
    staff_ currentStaff = {};

    int i = 0;
    while (i < linesOverThreshold.size()) {
        currentStaff.lines[lineCounter % 5] = line_ { linesOverThreshold[i] };

        // skip consecutive "lines" since they represent the same line
        i++;
        while (i < linesOverThreshold.size() && linesOverThreshold[i] == linesOverThreshold[i - 1] + 1) {
            i++;
        }

        // if all 5 lines found, save currentStaff and restart collecting
        lineCounter++;
        if (lineCounter % 5 == 0) {
            staffs.push_back(currentStaff);
            currentStaff = {};
        }
    }

    if (SHOW_STAFFS) {
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

    return dilationImg;
}


// "Opening consists of an erosion followed by a dilation and can be used to eliminate
//  all pixels in regions that are too small to contain the structuring element."
cv::Mat_<uchar> opening(cv::Mat_<uchar> img, const cv::Mat_<uchar>& sel) {
    cv::Mat_<uchar> imgAux = erosion(img, sel);
    cv::Mat_<uchar> imgRes = dilation(imgAux, sel);

    if (SHOW_OPENING) {
        imshow("Opening", imgRes);
    }

    return imgRes;
}


// Search for connected components in img using Breadth First Traversal
// Modified for the project's needs: it follows the reading direction of a music sheet so labeling comes "in order"
cv::Mat_<int> connectedComponentsBFS(cv::Mat_<uchar> img, const std::vector<staff_>& staffs, int &maxLabel) {
    int currentLabel = 0;						        // counter for labeling
    cv::Mat_<int> labelsImg(img.rows, img.cols, 0);	    // labels of corresponding pixels, initially all 0s (unlabeled)

    // index offsets for 8-neighborhood
    int di[8] = { -1, -1, -1, 0, 1, 1,  1,  0 };
    int dj[8] = { -1,  0,  1, 1, 1, 0, -1, -1 };

    int pi, pj;	 // pixel index
    int ni, nj;	 // neighbor index

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

        colors[0] = cv::Vec3b(255.0, 255.0, 255.0);         // we consider 0 unlabeled, i.e. background
        bool seenLabel[currentLabel + 1];
        for (int i = 1; i <= currentLabel; i++) {
            colors[i] = cv::Vec3b(d(gen), d(gen), d(gen));	// other labels have random color
            seenLabel[i] = false;
        }

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


// Draw a cross on image img, "around" point p, with given diameter (and optionally color)
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


// Extract a binary object from the labels image
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


// Get duration of a note
duration_ getDuration(cv::Mat_<uchar> img, cv::Point2i com, const cv::Mat_<uchar>& flagImg, std::vector<int> linesOverThreshold) {
    cv::Mat_<uchar> noLinesImg = opening(img, stemStructuringElement);

    if (SHOW_NO_LINE) {
        cv::imshow("No Line", noLinesImg);
    }

    // start of a new connected component (new BFS)
    std::queue<cv::Point2i> Q;
    Q.push(com);

    // index offsets for 8-neighborhood
    int di[8] = { -1, -1, -1, 0, 1, 1,  1,  0 };
    int dj[8] = { -1,  0,  1, 1, 1, 0, -1, -1 };

    int pi, pj;	// pixel index
    int ni, nj;	// neighbor index

    // a new canvas for extracting connected component starting from center of mass => the note with the stem
    cv::Mat_<uchar> compImg(img.rows, img.cols);
    for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
            compImg(i, j) = 255;
        }
    }

    while (!Q.empty()) {
        // dequeue and decompose
        cv::Point2i p = Q.front();
        pi = p.y;
        pj = p.x;
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
            if (noLinesImg(ni, nj) != 0 || compImg(ni, nj) == 0) {
                continue;
            }

            compImg(ni, nj) = 0;
            Q.push(cv::Point2i (nj, ni));
        }
    }

    // get new center of mass, so we know the direction the stem goes
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

        if (img(fy, fx) == 0) {
            drawCross(flagImg, cv::Point2i(fx, fy), 10, 50);
            return eighth;
        }

        fx = endPoint.x - xOffset;  // check to the left
        if (img(fy, fx) == 0) {
            drawCross(flagImg, cv::Point2i(fx, fy), 10, 50);
            return eighth;
        }

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

    return quarter;
}


std::vector<note_> extractNotes(const cv::Mat_<int>& binaryImg, cv::Mat_<int> labelImg, int maxLabel, std::vector<staff_> staffs, const std::vector<int>& linesOverThreshold) {
    // image to show each node head's center of mass (with drawCross)
    cv::Mat_<uchar> comImg = cv::Mat::zeros(labelImg.rows, labelImg.cols, CV_8UC1);

    // image to show flag/beam detection points (with drawCross)
    cv::Mat_<uchar> flagImg = copyImageWithGrayUchar(binaryImg);

    std::vector<int> noteLabels;
    std::vector<note_> notes;

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
        if (com.x < MIN_X_NOTE_HEAD) {
            continue;
        }
        if (SHOW_CENTER_OF_MASS) {
            drawCross(comImg, com, 50);
        }

        duration_ duration = getDuration(binaryImg, com, flagImg, linesOverThreshold);

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

    if (SHOW_CENTER_OF_MASS) {
        imshow("CenterOfMass", comImg);
    }

    if (SHOW_FLAGS) {
        cv::imshow("Flags", flagImg);
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


// Generate notes.txt
void writeNotesToFile(const std::vector<note_>& notes) {
    std::ofstream outFile;
    outFile.open("notes.txt");

    if (SHOW_NOTE_ENCODINGS) {
        std::cout << "Encoded notes:" << std::endl;
    }

    for (note_ n : notes) {
        std::string encodedNote = encodeNote(n);
        if (SHOW_NOTE_ENCODINGS) {
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
    std::vector<int> linesOverThreshold = getLinesOverThreshold(binaryImg,horizontalProjection);
    std::vector<staff_> staffs = getStaffs(binaryImg, linesOverThreshold);

    cv::Mat_<uchar> openingImg = opening(binaryImg, noteHeadStructuringElement);
    int maxLabel;
    cv::Mat_<int> labelImg = connectedComponentsBFS(openingImg, staffs, maxLabel);

    std::vector<note_> notes = extractNotes(binaryImg, labelImg, maxLabel, staffs, linesOverThreshold);
    writeNotesToFile(notes);

    if (RUN_PYTHON_SCRIPT) {
        system(PYTHON_COMMAND);
    }

    cv::waitKey(0);
    return 0;
}
