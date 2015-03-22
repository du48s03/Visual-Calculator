import Image
import numpy
from collections import deque
from subprocess import call
import ImageMath, ImageFilter, ImageDraw
import math
import re

def takeShot(filename="image"):
    call(["CommandCam", "/filename", filename+".bmp", "/quiet", "/delay", "1"])
    return Image.open(filename+".bmp")

def findPieces(image):
    pix = image.load()
    #visited = numpy.zeros([im.size[0], im.size[1]])
    pieceList = []
    unvisited = {(x,y) for x in range(image.size[0]) for y in range(image.size[1]) if pix[x,y] > 127}
    while(len(unvisited) != 0):
        queue = deque( [next(iter(unvisited))] ) 
        unvisited.remove(queue[0])
        piece = set()
        while(len(queue) != 0 ):
            p = queue.popleft()
            piece.add(p)
            neigh = neighbors(image.size, p)
            for n in neigh:
                if(pix[n[0], n[1]] < 127):
                    continue
                if(n in unvisited):
                    queue.append(n)
                    unvisited.remove(n) #indicating that this pixel is visited
        pieceList.append(piece)
    return pieceList

def find_void(image):
    pix = image.load()
    #visited = numpy.zeros([im.size[0], im.size[1]])
    pieceList = []
    unvisited = {(x,y) for x in range(image.size[0]) for y in range(image.size[1]) if pix[x,y] < 127}
    while(len(unvisited) != 0):
        queue = deque( [next(iter(unvisited))] ) 
        unvisited.remove(queue[0])
        piece = set()
        while(len(queue) != 0 ):
            p = queue.popleft()
            piece.add(p)
            neigh = neighbors(image.size, p, diag=True)
            for n in neigh:
                if(pix[n[0], n[1]] > 127):
                    continue
                if(n in unvisited):
                    queue.append(n)
                    unvisited.remove(n) #indicating that this pixel is visited
        pieceList.append(piece)
    return pieceList

def find_holes(image, PieceList, min_hole_size=200):
    ret = list(PieceList)
    size = image.size
    for piece in PieceList:
        if(len(piece) < min_hole_size):
            ret.remove(piece)
            continue
        for pix in piece:
            if(pix[0] == 0 or pix[0] == size[0]-1 or pix[1] == 0 or pix[1] == size[1]-1):
                ret.remove(piece)
                break
    return ret

def findHand(pieceList):
    return reduce(lambda x, y:x if len(x)>len(y) else y, pieceList)

def neighbors(size, pos, diag=False):
    """order: left, up, right, down"""
    ret = []
    i,j = pos[0], pos[1]
    if(i>0):
        ret.append((i-1,j))
    if(j>0):
        ret.append((i,j-1))
    if(i<size[0]-1):
        ret.append((i+1, j))
    if(j<size[1]-1):
        ret.append((i,j+1))
    if(diag):
        if(i>0 and j>0):
            ret.append((i-1,j-1))
        if(i < size[0]-1 and j>0):
            ret.append((i+1,j-1))
        if(i<size[0]-1 and j < size[1]-1):
            ret.append((i+1, j+1))
        if(i>0 and j<size[1]-1):
            ret.append((i-1,j+1))
    return ret

def norm_color(image):
    out=Image.new("RGB", image.size, None)
    outpix = out.load()
    inpix = image.load()
    norm_length=127*3**0.5
    for i in range(out.size[0]):
        for j in range(out.size[1]):
            norm = (float(inpix[i,j][0])**2+float(inpix[i,j][1])**2+float(inpix[i,j][2])**2)**0.5
            if (norm == 0):
                outpix[i,j]=(127, 127, 127)
            else:
                outpix[i,j]=(int(inpix[i,j][0]/norm*norm_length), 
                    int(inpix[i,j][1]/norm*norm_length), 
                    int(inpix[i,j][2]/norm*norm_length))
    return out

def find_edgepoints(image, piece):
    edgepoints = set()
    pix = image.load()
    for pos in piece:
        neigh = neighbors(image.size, pos, diag=True)
        for n in neigh:
            if(pix[n[0], n[1]] < 127 and n not in edgepoints):
                edgepoints.add(n)
    return edgepoints

def find_edges(image, edgepoints):
    unvisited=set(edgepoints)
    done_visited = set()
    ret = []
    while(len(unvisited) != 0):
        p = next(iter(unvisited))
        edge = []
        queue = deque([p])
        unvisited.remove(p)
        while(len(queue) != 0):
            pos = queue.popleft()
            edge.append(pos)
            neigh = neighbors(image.size, pos, diag=True)
            for n in neigh:
                if(n not in edgepoints):
                    continue
                if(n in unvisited):
                    queue.append(n)
                    unvisited.remove(n)
                else:#cycle detected
                    pass
         
        # find_edge_dfs(image, edgepoints, p, visited, done_visited, edge)
        ret.append(edge)
    return ret

def find_edge_dfs(image, edgepoints, point, visited, done_visited, edge):
    neigh = neighbors(image.size, point)
    for n in neigh:
        if(n in edgepoints and n not in visited):
            edge.add(n)
            visited.add(n)
            if(n not in done_visited): # a cycle is detected
                pass
            find_edge_dfs(image, edgepoints, n, visited, done_visited, edge)
    done_visited.add(point)

def skin_mask(image):
    source = image.convert('RGB').split()
    redmask = source[0].point(lambda i: i> 10 and 255)
    return redmask

def drawedges(image, edges, name='edges.jpg'):
    img = Image.new('L', image.size, 'black')
    pix = img.load()
    for i in range(len(edges)):
        for p in edges[i]:
            # len(edges)+1
            # 255/(len(edges)+1)
            # round(255/(len(edges)+1))
            # i*round(255/(len(edges)+1))
            # 255-i*round(255/(len(edges)+1))
            pix[p[0],p[1]] = int(255-i*round(255/(len(edges)+1)))
    img.save(name)

def fitness(fingermask, pos, fil, size=None):
    """pos is the upper-left corner"""
    crop = fingermask.crop((pos[0], pos[1], pos[0]+fil.size[0], pos[1]+fil.size[1] ))
    if(ImageMath.eval("crop and crop", crop=crop) is False):
        return 0
    crop = crop.convert("I").point(lambda i: i * 0.00785+(-1))
    conv = ImageMath.eval("crop*fil", crop=crop, fil=fil)
    s = 0
    pix = conv.load()
    for i in range(crop.size[0]):
        for j in range(crop.size[1]):
            # print "i=$i, j=$j, conv=$pix[$i,$j]"
            s+=pix[i,j]
    return s


def finger_filter(size=(100,300)):
    fil = Image.new('F', size, 'black')
    pix = fil.load()
    yt = float(fil.size[1]/4)
    xt = float(fil.size[0]/2)
    sigmax = float(fil.size[0]/4)
    sigmay = float(sigmax*2)
    intensity = 1
    norm = 0
    for i in range(fil.size[0]):
        for j in range(fil.size[1]):
            if ( j > yt):
                pix[i,j] = intensity*math.exp(-(i-xt)**2/(sigmax**2))
            else:
                pix[i,j] = intensity*math.exp(-(i-xt)**2/(sigmax**2)-(j-yt)**2/(sigmay**2))
            norm += pix[i,j]
    for i in range(fil.size[0]):
        for j in range(fil.size[1]):
            pix[i,j] = pix[i,j]-norm/float(fil.size[0]*fil.size[1])
    return fil



def find_finger(skinmask):
    """only search along the horizontal line at top-most nonzero pixel"""
    #Can go much faster if we compute the fitness function directly
    if skinmask.filter(ImageFilter.ModeFilter).getbbox() is None:
        return None
    fingertip_y = skinmask.filter(ImageFilter.ModeFilter).getbbox()[1]+1
    fingertip_x=[]
    for i in range(skinmask.size[0]):
        if(skinmask.getpixel((i, fingertip_y))>0):
            fingertip_x.append(i)
    max_fitness = None
    finger_pos = None
    ff = finger_filter()
    checkset = reduce(lambda x,y: x.union(y),
                    map(lambda x:set(range(x-ff.size[0], x)), fingertip_x))
    for i in checkset:
        f = fitness(skinmask, (i, fingertip_y), ff)
        # if(f != 0):
        #     print 'x = ', i, 'y = ', fingertip, 'f = ', f
        if(max_fitness is None or f > max_fitness):
            max_fitness= f
            finger_pos = i
    # if(finger_pos is not None):
    return (max_fitness, (finger_pos, fingertip_y, finger_pos+ff.size[0], fingertip_y+ff.size[1]))

def horizontal_palm(size=(450, 200)):
    fil = Image.new('F', size, 'black')
    pix = fil.load()
    yt = float(fil.size[1]/2)
    xt = float(fil.size[0]/3)
    sigmay = float(fil.size[1]/4)
    sigmax = float(fil.size[0]*0.6)
    intensity = 1
    norm = 0
    for i in range(fil.size[0]):
        for j in range(fil.size[1]):
            if ( i < xt):
                pix[i,j] = intensity*math.exp(-(j-yt)**2/(sigmay**2))
            else:
                pix[i,j] = intensity*math.exp(-(i-xt)**2/(sigmax**2)-(j-yt)**2/(sigmay**2))
            norm += pix[i,j]
    for i in range(fil.size[0]):
        for j in range(fil.size[1]):
            pix[i,j] = pix[i,j]-norm/float(fil.size[0]*fil.size[1])
    return fil    

def find_horizontal(skinmask):
    """only search along the vertical line at right-most nonzero pixel"""
    if skinmask.filter(ImageFilter.ModeFilter).getbbox() is None:
        return None
    fingertip_x = skinmask.filter(ImageFilter.ModeFilter).getbbox()[2]-1
    fingertip_y=[]
    for j in range(skinmask.size[1]):
        if(skinmask.getpixel((fingertip_x, j))>0):
            fingertip_y.append(j)
    max_fitness = None
    finger_pos = None
    ff = horizontal_palm()
    checkset = reduce(lambda x,y: x.union(y),
                    map(lambda x:set(range(x-ff.size[0], x)), fingertip_y))
    for j in checkset:
        f = fitness(skinmask, (fingertip_x-ff.size[0], j), ff)
        if(max_fitness is None or f > max_fitness):
            max_fitness= f
            finger_pos = j
    return (max_fitness, (fingertip_x-ff.size[0], finger_pos, fingertip_x, finger_pos+ff.size[1]))


def extract_feat(image, filename="image"):
    ret = {}
    skinmask = skin_mask(image)
    fh = re.split('\.', filename)
    skinmask.save(fh[0]+'_skinmask.jpg')

    ret["bbox"]=skinmask.filter(ImageFilter.ModeFilter).getbbox()
    if ret["bbox"] is None:
        return ret

    #---- finding holes
    black_pieces = find_void(skinmask)
    print "num of black pieces = ", len(black_pieces)
    holes = find_holes(skinmask, black_pieces)
    ret["hole_number"] = len(holes)
    if(len(holes) != 0):
        max_hole = None
        max_hole_size = 0
        max_hole_ind = None
        for i in range(len(holes)):
            if(max_hole_size < len(holes[i])):
                max_hole = holes[i]
                max_hole_size = len(holes[i])
                max_hole_ind = i
        max_hole_left = min(max_hole, key=lambda i:i[0])[0]
        max_hole_right = max(max_hole, key=lambda i:i[0])[0]
        max_hole_top = min(max_hole, key=lambda i:i[1])[1]
        max_hole_bottom = max(max_hole, key=lambda i:i[1])[1]
        ret["biggest_hole"]=(max_hole_left, max_hole_top, max_hole_right, max_hole_bottom)
        ret["biggest_hole_size"]=max_hole_size
    

    #---- finding the vertical finger
    tmp = find_finger(skinmask)
    # if(tmp is not None):
    max_fingerness = tmp[0]
    finger_pos = tmp[1]
    # if(max_fingerness > fingerness_thres):
    ret["max_fingerness"]=max_fingerness
    ret["finger_pos"]=finger_pos

    #---- finding the horizontal palm
    tmp = find_horizontal(skinmask)
    # if(tmp is not None):
    max_hori_fit = tmp[0]
    hori_finger_pos = tmp[1]
    # if(max_hori_fit > hori_fit_thres):
    ret["max_hori_fit"]=max_hori_fit
    ret["hori_finger_pos"]=hori_finger_pos

    return ret


def parse_features(features):
    fingerness_thres = 3000
    hori_fit_thres = 20000
    if(features["hole_number"]==1):
        # is a number
        if(features["max_fingerness"] < fingerness_thres):
            #number equals 0
            return '0'
        else:
            #number equals 1
            return '1'
    elif(features["hole_number"]==0):
        # is a symble
        if(features["max_fingerness"] > fingerness_thres and features["max_hori_fit"] <= hori_fit_thres):
            # is a 'plus'
            return '+'
        elif(features["max_hori_fit"] > hori_fit_thres and features["max_fingerness"] <= fingerness_thres):
            # is a 'equal'
            return '='
        else:
            #unknown
            return 'unknown'

    else:
        # unknown
        return 'unknown'

def drawboxes(skinmask, features, filename='image'):
    #--------for documentation
    fingerness_thres = 3000
    hori_fit_thres = 20000
    canvas = skinmask.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    if(features["hole_number"]>0):
        draw.rectangle(features["biggest_hole"], outline="red")
    if("max_fingerness" in features and features["max_fingerness"] > fingerness_thres):
        draw.rectangle(features["finger_pos"], outline="blue")
    if("max_hori_fit" in features and features["max_hori_fit"] > hori_fit_thres):
        draw.rectangle(features["hori_finger_pos"], outline="green")
    canvas.save(filename+'_box.jpg')
    OUT = open(filename+".txt", 'w')
    for keys in features:
        OUT.write(keys + ":"+str(features[keys])+'\n')
    OUT.write("\nclassification result = '"+parse_features(features)+"'\n")



if __name__ == "__main__":
    import sys
    import time
    stop=False
    seqindex = 0
    sleeptime = 3 #time in secs before the next picture is taken and proccessed
    #---- saving picture with fitting results
    state = 0 # 0 = number1, 1 = number2 
    s1 = 0
    s2 = 0
    expr = ""
    while(not stop):
        st = sleeptime
        while(st != 0):
            print "\rTaking next input in "+str(st)+" secs..\t"+expr, 
            time.sleep(1)
            st -= 1
        filename = sys.argv[1]+str(seqindex)
        image = takeShot(filename=filename)
        skinmask = skin_mask(image)
        features = extract_feat(image, filename=filename)
        if features["bbox"] is None:
            continue
        symble = parse_features(features)
        drawboxes(skinmask, features, filename=filename)  # for documentation
        if(symble is '0'):
            expr = expr + "0"
            s2 = s2*2
        elif(symble is '1'):
            expr = expr + "1"
            s2 = s2*2+1
        elif(symble is '+'):
            expr = expr + "+" 
            s1 += s2
            s2 = 0
        elif(symble is '='):
            expr = expr + "="
            s1 += s2
            s2 = 0
            stop = True
        else:
            pass
        seqindex+=1
    print "Ans = ", s1