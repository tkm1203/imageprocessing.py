import numpy as np
import os
import cv2
import sys
from matplotlib import pyplot as plt
from pylsd.lsd import lsd
import copy
import glob

IMAGE_PATH = "./dataset/images/" #元画像のあるディレクトリへのパス
GREEN_IMAGE_PATH = "./dataset/green_images/" #緑領域を抽出した画像のあるディレクトリへのパス
BINARY_IMAGE_PATH = "./dataset/binary_images/" #二値化画像のあるディレクトリへのパス
DIFF_IMAGE_PATH = "./dataset/diff_images/" #差分画像のあるディレクトリへのパス
HOUGHLINE_IMAGE_PATH = "./dataset/houghlines_images/" #ハフ変換画像のあるディレクトリへのパス
LSD_IMAGE_PATH = "./dataset/lsd_images/" #LSD変換画像のあるディレクトリへのパス
CANNY_IMAGE_PATH = "./dataset/canny_images/" #canny法を適用した画像のあるディレクトリへのパス
EDGE_DENSITY_IMAGE_PATH = "./dataset/edge_density_images" #エッジ強度分布画像のあるディレクトリへのパス
MOVIE_PATH = "./dataset/movies/" #トラッキング動画などの動画のあるディレクトリへのパス

# 緑色の検出
def detect_green_color(iamge):
    # HSV色空間に変換
    hsv = cv2.cvtColor(iamge, cv2.COLOR_BGR2HSV)

    # 緑色のHSVの値域1
    hsv_min = np.array([30, 45, 0])
    hsv_max = np.array([90,255,255])

    # 緑色領域のマスク（255：赤色、0：赤色以外）    
    mask = cv2.inRange(hsv, hsv_min, hsv_max)
    
    # マスキング処理
    masked_img = cv2.bitwise_and(iamge, iamge, mask=mask)

    return masked_img

# 確率的ハフ変換で直線を抽出する関数
def hough_lines_p(image,outLineImage):
    resultP = image
    # 確率的ハフ変換で直線を抽出
    lines = cv2.HoughLinesP(outLineImage, rho=1, theta=np.pi/180, threshold=80, minLineLength=150, maxLineGap=50)
    #print("hough_lines_p: ", len(lines))
  
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(resultP,(x1,y1),(x2,y2),(0,255,0),1) # 緑色で直線を引く
    
    return resultP

#pylsdによる線分検出
def LineSegmentedDetector(image,gray):
    resultL = image
    #image_copy = image.copy()
    linesL = lsd(gray)
    #print("lines_lsd:", len(linesL))
    for line in linesL:
        x1, y1, x2, y2 = map(int,line[:4])
        resultL = cv2.line(image, (x1,y1), (x2,y2), (0,0,255), 1)
        #if (x2-x1)**2 + (y2-y1)**2 > 1000:
            # 赤線を引く
        #    cv2.line(resultL, (x1,y1), (x2,y2), (0,0,255), 3)
    return resultL

#二値化関数
def binarization(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)

    return th

#リサイズ関数(第二引数は縮小(拡大)割合)
def resize(image,ratio):
    height = image.shape[0]
    width = image.shape[1]
    resize_image = cv2.resize(image,(int(width*ratio), int(height*ratio)))

    return resize_image

#画像差分を得るための関数(第一引数に変化後、第二引数に変化前の画像を入力)
def image_diff(image1,image2,choice):
    if image1.shape == image2.shape:
        #ndarrayを利用した方法とopencvを利用した方法を選択
        img_diff = image1.astype(int) - image2.astype(int)
        gray1 = cv2.cvtColor(image1,cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2,cv2.COLOR_BGR2GRAY)
        #gray_diff = gray1.astype(int) - gray2.astype(int)
        if choice == '0':
            diff_image =  np.abs(img_diff) #単純な差分画像
        if choice == '1':
            diff_image =  np.floor_divide(img_diff, 2) + 128 #差分の無いところを128に(グレー画像に)
        if choice == '2':
            diff_image =  (np.abs(img_diff) != 0) * 255  #差分を二値化して強調する
        if choice == '3':
            #diff_image =  cv2.absdiff(image1,image2)
            diff_image =  cv2.absdiff(gray1,gray2)

    else:
        print("The shape of these images are different!")
    
    return diff_image

#エッジ強度を求める関数
def edge_density(image,kernel):
    image2= cv2.imread(image)
    img = image2
    #print(img.shape)
    start = int(kernel/2)+1
    result = np.zeros((540+start,960+start,3),dtype=np.uint8)
    #sum_result = np.zeros((540+start,960+start,3),dtype=np.uint8)


    print(np.sum(image2[10:10+kernel,90:90+kernel]))

    for h in np.arange(image2.shape[0]):
        for w in np.arange(image2.shape[1]):
            sum_result = np.sum(image2[h:h+kernel,w:w+kernel])
            result[start+h,start+w] = sum_result
            
            #sum_result[start+h,start+w] = np.sum(image2[h:h+kernel,w:w+kernel])
            #print(sum_result[start+h,start+w])
        
    max = sum_result[np.unravel_index(np.argmax(sum_result), sum_result.shape)] #配列内の最大値
    print(np.unravel_index(np.argmax(sum_result), sum_result.shape))
    #result = int(120/max * result) + 30
    print(max)
    #print(sum_result[13][91])
    result =  sum_result
    print(result.shape)
    #result[:,:,1] = int(255/max * result[:,:,1])
    #result[:,:,2] = int(255/max * result[:,:,2])
    cv2.imwrite(EDGE_DENSITY_IMAGE_PATH + "image001.jpg",result)
    return result

def get_tracked_movie(videofile,width,height):
    filename = os.path.basename(videofile)
    tracker = select_tracker()
    tracker_name = str(tracker).split()[0][1:]

    frame_rate = int(input("フレームレートを入力: "))

    cap = cv2.VideoCapture(videofile)
    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    video = cv2.VideoWriter(MOVIE_PATH + 'tracked_' + filename, fourcc, frame_rate, (width, height))
    #webカメラの軌道に時間がかかる場合
    import time
    time.sleep(1)

    ret, frame = cap.read()

    roi = cv2.selectROI(frame, False)

    ret = tracker.init(frame, roi)

    while True:

        ret, frame = cap.read()

        success, roi = tracker.update(frame)

        (x,y,w,h) = tuple(map(int,roi))

        if success:
            p1 = (x, y)
            p2 = (x+w, y+h)
            cv2.rectangle(frame, p1, p2, (0,255,0), 3)
        else :
            cv2.putText(frame, "Tracking failed!!", (500,400), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),3)
        
        #fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
        #video = cv2.VideoWriter('./images/tracked.mp4', fourcc, frame_rate, (1920, 1080))

        video.write(frame)
        cv2.imshow(tracker_name, frame)
        
        k = cv2.waitKey(1) & 0xff
        if k == 27 :
            break
    
    video.release()
    cap.release()
    cv2.destroyAllWindows()

# トラッカーを選択する
def select_tracker():
    print("Which Tracker API do you use?")
    print("0: Boosting")
    print("1: MIL")
    print("2: KCF")
    print("3: TLD")
    print("4: MedianFlow")
    choice = input("Please select your tracker number: ")

    if choice == '0':
        tracker = cv2.TrackerBoosting_create()
    if choice == '1':
        tracker = cv2.TrackerMIL_create()
    if choice == '2':
        tracker = cv2.TrackerKCF_create()
    if choice == '3':
        tracker = cv2.TrackerTLD_create()
    if choice == '4':
        tracker = cv2.TrackerMedianFlow_create()

    return tracker

#画像ファイルから一括で緑領域を抽出する関数
def get_green_image(files):
    

    for original_image in files:
        filename = os.path.basename(original_image)
        image = cv2.imread(original_image)
        green_image = detect_green_color(image)
        cv2.imwrite(GREEN_IMAGE_PATH + filename,green_image)
        print("{} ---> get green image".format(filename))

#2つの画像ファイルを一括でリサイズする関数
def get_resized_image(files1,files2,ratio):
    for original,green in zip(files1,files2):

        filename_original = os.path.basename(original)
        filename_green = os.path.basename(green)

        original_image = cv2.imread(original)
        green_image = cv2.imread(green)

        original_resize = resize(original_image,ratio)
        green_resize = resize(green_image,ratio)

        cv2.imwrite(IMAGE_PATH + filename_original,original_resize)
        cv2.imwrite(GREEN_IMAGE_PATH + filename_green,green_resize)

        print("{} ---> get resized image".format(filename_original))

#画像ファイルを一括で二値化する関数
def get_binarized_image(files):
    if not os.path.isdir(BINARY_IMAGE_PATH):
        os.mkdir(BINARY_IMAGE_PATH)

    for green in files:
        filename = os.path.basename(green)
        green_image = cv2.imread(green)
        binary_image = binarization(green_image)
        cv2.imwrite(BINARY_IMAGE_PATH + filename, binary_image)
        print("{} ---> get binarized image".format(filename))

def get_diff_image(files):
    if not os.path.isdir(DIFF_IMAGE_PATH):
        os.mkdir(DIFF_IMAGE_PATH)

    print("which diff method do you use?")
    print("0:単純な差分画像")
    print("1:差分0をグレースケールに")
    print("2:差分を二値化")
    print("3:opencvを用いた差分画像")
    choice = input("Please select your diff method number: ")

    for i in range(len(files)-1):
        image1 = cv2.imread(files[i])
        image2 = cv2.imread(files[i+1])
        diff = image_diff(image2,image1,choice)
        cv2.imwrite(DIFF_IMAGE_PATH + "diff_" + str(i+1) + ".jpg",diff)
        print("diff of image{}-image{} have got".format(str(i+2),str(i+1)))
    
def get_houghline_image(files1,files2):
    if not os.path.isdir(CANNY_IMAGE_PATH):
        os.mkdir(CANNY_IMAGE_PATH)
    
    if not os.path.isdir(HOUGHLINE_IMAGE_PATH):
        os.mkdir(HOUGHLINE_IMAGE_PATH)

    for images,greens in zip(files1,files2):
        filename = os.path.basename(images)
        image = cv2.imread(images)
        green = cv2.imread(greens)

        gray = cv2.cvtColor(green,cv2.COLOR_BGR2GRAY)
        outLineImage = cv2.Canny(gray, 200, 200, apertureSize = 3) # 輪郭線抽出
        cv2.imwrite(CANNY_IMAGE_PATH + filename, outLineImage)
        print("{} ---> get canny image".format(filename))

        hough_image = hough_lines_p(image,outLineImage)
        cv2.imwrite(HOUGHLINE_IMAGE_PATH + filename,hough_image)
        print("{} ---> get hough lines image".format(filename))

def get_LSD_image(files1,files2):
    if not os.path.isdir(LSD_IMAGE_PATH):
        os.mkdir(LSD_IMAGE_PATH)

    for images,greens in zip(files1,files2):
        filename = os.path.basename(images)
        image = cv2.imread(images)
        green = cv2.imread(greens)
        gray = cv2.cvtColor(green,cv2.COLOR_BGR2GRAY)

        lsd = LineSegmentedDetector(image,gray)
        cv2.imwrite(LSD_IMAGE_PATH + filename,lsd)

        print("{} ---> get LSD image".format(filename))

#画像を動画に変換
def get_movie(images,width,height,frame_rate):
    if not os.path.isdir(MOVIE_PATH):
        os.mkdir(MOVIE_PATH)

    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    video = cv2.VideoWriter(MOVIE_PATH + 'images.mp4', fourcc, frame_rate, (width, height))
   
    print("動画変換中...")
   
    for i in range(len(images)):
        img = cv2.imread(images[i])
        #img = cv2.resize(img,(1920,1080))
        video.write(img) 
       
    video.release()
    print("動画変換完了")

def get_edge_density_image(files,kernel):

    SAVEPATH = EDGE_DENSITY_IMAGE_PATH + "_kernel" + str(kernel) + "/"

    if not os.path.isdir(SAVEPATH):
        os.mkdir(SAVEPATH)

    for images in files:
        filename = os.path.basename(images)
        edge_image = cv2.imread(images)

        edge_density_image = edge_density(edge_image,kernel)

        cv2.imwrite(SAVEPATH + filename,edge_density_image)

        print("{} ---> get edge density image".format(filename))

if __name__ == "__main__":

    if not os.path.isdir(GREEN_IMAGE_PATH):
        os.mkdir(GREEN_IMAGE_PATH)
        
    image_files = sorted(glob.glob(IMAGE_PATH + "*.jpg"))
    green_image_files = sorted(glob.glob(GREEN_IMAGE_PATH + "*.jpg"))
    #動画のサイズ
    width = 960
    height = 540

    print("やりたい操作を選択")
    print("0:緑色領域抽出")
    print("1:画像の一括リサイズ")
    print("2:画像を二値化")
    print("3:画像差分を取得")
    print("4:ハフ変換画像を取得")
    print("5:LSDによる線分検出画像を取得")
    print("6:画像から動画を作成")
    print("7:エッジ強度分布を求める")
    print("8:トラッキング映像の取得")
    print("9:プログラムの終了")
    choice = input("行う操作の番号: ")
    print("")
    
    
    if choice == '0':
        get_green_image(image_files) #緑色領域を抽出
    if choice == '1':
        ratio = int(input("リサイズする割合を入力: "))
        get_resized_image(image_files,green_image_files,ratio) #画像をリサイズ
    if choice == '2':
        get_binarized_image(green_image_files) #画像を二値化
    if choice == '3':
        get_diff_image(green_image_files) #画像差分を取得
    if choice == '4':
        get_houghline_image(image_files,green_image_files) #ハフ変換画像を取得
    if choice == '5':
        get_LSD_image(image_files,green_image_files) #LSDによる線分検出画像を取得
    if choice == '6':
        frame_rate = int(input("動画のフレームレートを入力: "))
        get_movie(image_files,width,height,frame_rate) #画像から動画を作成
    if choice == '7':
        edge_image_files = sorted(glob.glob(CANNY_IMAGE_PATH + "*.jpg"))
        kernel = int(input("カーネルサイズを入力: "))
        edge_density(CANNY_IMAGE_PATH + "image001.jpg",kernel)    
        #get_edge_density_image(edge_image_files,kernel) #エッジ強度分布を求める
    if choice == '8':
        get_tracked_movie(MOVIE_PATH + "images.mp4",width,height) #映像からトラッキングを行う
    if choice == '9':
        print("finish program")
        sys.exit(0)