import tensorflow as tf
import sys, os, gc

from TF2DeepFloorplan.dfp.net import *
from TF2DeepFloorplandfp.data import *
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from argparse import Namespace
from datetime import datetime, timezone
import pytz
import pandas as pd
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
sys.path.append('./TF2DeepFloorplan/dfp/utils/')
from TF2DeepFloorplan.dfp.utils.rgb_ind_convertor import *
from TF2DeepFloorplan.dfp.utils.util import *
from TF2DeepFloorplan.dfp.utils.legend import *
from TF2DeepFloorplan.dfp.deploy import *
sys.path.append('./TF2DeepFloorplan/')
sys.path.append('./TF2DeepFloorplan/dfp')

fPlan_link = sys.argv[0]
inp = mpimg.imread(str(fPlan_link))
args = Namespace(image='./TF2DeepFloorplan/resources/30939153.jpg', weight='./log/store/G', loadmethod='log', postprocess=True, colorize=True, save=None)
result = main(args)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                    FLOOR PLAN EXTRACTION
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

# Seperate room spaces and walls
model,img,shp = init(args)
logits_cw,logits_r = predict(model,img,shp)
logits_r = tf.image.resize(logits_r,shp[:2])
logits_cw = tf.image.resize(logits_cw,shp[:2])
r = convert_one_hot_to_image(logits_r)[0].numpy()
cw = convert_one_hot_to_image(logits_cw)[0].numpy()
r_color,cw_color = colorize(r.squeeze(),cw.squeeze())
newr,newcw = post_process(r,cw,shp)
newr_color, newcw_color = colorize(newr.squeeze(),newcw.squeeze())

# Converting images to gray scale for later processing
newcw_color = np.asarray(newcw_color, dtype='u1')
gray = cv2.cvtColor(newcw_color, cv2.COLOR_BGR2GRAY)

#Checkpoint
cp1 = newcw_color.copy()
newcw_color = cp1.copy()

########################### GETTING ROOM SPACES ###########################
###########################################################################

ret,thresh = cv2.threshold(gray,1,255,0)
contours,hierarchy = cv2.findContours(thresh, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

softConerL = cnt_to_softCnr(contours)

sortedCnt_Area = sort_contour_area(softConerL)
sortedCnts = sortedCnt_Area[:,1]
boxImg, bboxes = draw_bounding_box(sortedCnts,boxThreshold=len(sortedCnts))


########################### GETTING WINDOWS/ DOORS ###########################
##############################################################################

# Filter the window or door with the key color = 103
wdFilter = (np.array(gray) == 103).astype(int)
wdOnly = np.array((wdFilter*255),dtype= 'u1')
wdGray = cv2.cvtColor(wdOnly, cv2.COLOR_GRAY2BGR)
wdContours, wdHierarchy = cv2.findContours(wdOnly, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
sWdCnt_Area = sort_contour_area(wdContours)
sWdCnt = sWdCnt_Area[:,1]
boxWdImg, WdBboxes = draw_bounding_box(sWdCnt,boxThreshold=len(sWdCnt))


########################### EXPORTING RESULTS ###########################
##############################################################################

wallLength_by_room = wall_length_cal(WdBboxes,sortedCnts,bboxes)
roomPerimeter = [round(cv2.arcLength(cnt, True),4) for cnt in sortedCnts]
roomArea =[round(cv2.contourArea(cnt),4) for cnt in sortedCnts]
rooms = pd.DataFrame(data=np.column_stack((range(1, len(roomArea) + 1),roomArea, roomPerimeter, wallLength_by_room)),columns=['Room_No,','Area', 'Perimeter', 'Wall_Length'])

# Exporting  the files
planName = planName = os.path.splitext(str(link))[0].split('/')[-1]
rooms.to_csv( str(planName) + '_' + getCTimeStr() + '.csv')

imgShow = inp.copy()
for idx, cnt in enumerate(sortedCnts):
  area = roomArea[idx]
  perimeter = roomPerimeter[idx]
  wallLength = wallLength_by_room[idx]

  x1, y1 = cnt[cnt[:, 0].argsort()][0,0], cnt[cnt[:, 1].argsort()][0,1]
  cv2.putText(imgShow, f'Room {int(idx) + 1}', (x1+6, y1+15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0,120,212), 1)

output = Image.fromarray(imgShow, 'RGB')
output = output.save(str(planName) + '_with_room_number.jpg')
plt.figure(figsize = (30,20))
plt.imshow(imgShow, cmap='gray')
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                                    HELPER FUNCTIONS
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

########################### PROCESSING AND REDUCING THE CORNERS ON CONTOURS ###########################

### Getting soft conner contour
def cnt_to_softCnr(contours):
  se_Cnt = []
  for cnt in contours:
    newcw_color = cp1.copy()
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    perimeter = round(perimeter, 4)

    epsilon = 0.004*cv2.arcLength(cnt,True)

    se_Cnt.append(np.array([cv2.approxPolyDP(cnt, epsilon,True)])[0,:,0])
  
  return se_Cnt

def sort_contour_area(contours):
    ### Calculate and create a descending sorted list of contour areas
    cnt_area = []
    for cnt in contours:
      # Calculate the area of the contour
      cnt_area.append([cv2.contourArea(cnt), cnt])

    cnt_area = np.array(cnt_area)
    sortedArr = cnt_area[cnt_area[:, 0].argsort()][::-1]

    return sortedArr

def draw_bounding_box(contours, image = None, boxThreshold=1):
    # Call our function to get the list of contour areas
    # sortedCntList = sort_contour_area(contours)
    bndBoxes = []
    # Loop through each contour of our image

    cntL =  contours if boxThreshold >= len(contours) else contours[:boxThreshold] 

    for cnt in cntL:
             
        # Use OpenCV boundingRect function to get the details of the contour
        x,y,w,h = cv2.boundingRect(cnt)
        bndBoxes.append(np.array([x, y, x+w, y+h]))
        # Draw the bounding box
        image=cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),2)
 
    return image, bndBoxes


########################### CHECKING WINDOWS/ DOORS AND WALL CONTACTING ###########################

# Binary search 
def bSearch(arr, low, high, x):
    """
    arr: array of pixel x of top left corners
    low: lower pointer (index)
    high: higher pointer (index)
    x: insert value for comparision
    -------------------------------
    RETURN the index that x just abit larger than arr[index].
    """
    if arr[low] > x:
        return -1
    
    if arr[high] < x:
        return high

    if high >= low:

        mid = (high + low) // 2

        if arr[mid] == x or (x < arr[mid+1] and x > arr[mid]):
            return mid
        
        elif arr[mid] > x:
            return bSearch(arr, low, mid-1, x)

        elif arr[mid] < x:
            return bSearch(arr, mid+1, high, x)

    else:
        return -1


def isInside(obj, box):
    """
    obj: pixel location of tree (x, y)
    box: bounding box - 2 pixel points of top left and bottom right corners (x1,y1,x2,y2)
    """

    oX, oY = obj
    tX, tY, bX, bY = box 

    x_inRange = oX >= tX and oX <= bX
    y_inRange = oY >= tY and oY <= bY

    return x_inRange and y_inRange


def get_recessive_side(box, rcs=True):
  X1, Y1, X2, Y2 = box
  print
  X_ = abs(X2-X1)
  Y_ = abs(Y2-Y1)

  rcsD = min(X_,Y_)
  domD = max(X_,Y_)

  return  rcsD if rcs else domD


def inErrorRange(pt, err, room):
    x, y = pt
    rX1, rY1, rX2, rY2 = room
    return abs(x-rX1) <= err or abs(x-rX2) <= err or abs(y-rY1) <= err or abs(y-rY2) <= err


def isContacted(wd, room):
    wdX1, wdY1, wdX2, wdY2 = wd
    error_ = get_recessive_side(wd)//2
    rX1, rY1, rX2, rY2 = room
    ##Extend error rate for window (4 corners)
    tl = (wdX1, wdY1)
    tr = (wdX1, wdY2)
    bl = (wdX2, wdY1)
    br = (wdX2, wdY2)

    return (inErrorRange(tl,error_,room) or inErrorRange(tr,error_,room) or inErrorRange(bl,error_,room) or inErrorRange(br,error_,room))

def find_contactedWd(WdBboxL, room):
  contacted = []
  for wd in WdBboxL:
    if isContacted(wd, room):
      contacted.append(wd)

  return contacted

def wall_length_cal(WdBboxes, RCnts ,RBboxes):
  wallLength = []
  
  for idx, room in enumerate(RBboxes):
    wdList =  find_contactedWd(WdBboxes, room)
    total_wdLength = 0
    if len(wdList) != 0:
      for wd in wdList:
        total_wdLength += get_recessive_side(wd,False)

    room_perimeter = round(cv2.arcLength(RCnts[idx], True),4)
    wallLength.append(round(room_perimeter - total_wdLength,4))   

  return wallLength


########################### EXTRAS ###########################

# Timestamp
def getCTimeStr():
  py_timezone = pytz.timezone('Australia/Adelaide')
  dt_string = datetime.now(py_timezone).strftime("(%H%M%S-%b%d%Y)")
  # print("date and time =", dt_string)	
  return str(dt_string)