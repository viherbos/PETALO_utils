import json
import os
import numpy as np
import sys
import pandas as pd
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.opengl.GLGraphicsItem import GLGraphicsItem
from pyqtgraph.Qt import QtCore, QtGui
sys.path.append("../PETALO_DAQ_infinity/")
from SimLib import config_sim as conf
from SimLib import DAQ_infinity as DAQ



class GLTextItem(GLGraphicsItem):
    def __init__(self, X=None, Y=None, Z=None, text=None, size=7, color=[50,50,50]):
        GLGraphicsItem.__init__(self)
        self.text = text
        self.X = X
        self.Y = Y
        self.Z = Z
        self.size = size
        self.color = color

    def setGLViewWidget(self, GLViewWidget):
        self.GLViewWidget = GLViewWidget

    def setText(self, text):
        self.text = text
        self.update()

    def setX(self, X):
        self.X = X
        self.update()

    def setY(self, Y):
        self.Y = Y
        self.update()

    def setZ(self, Z):
        self.Z = ZrenderText
        self.update()

    def paint(self):
        self.GLViewWidget.qglColor(QtGui.QColor(*self.color))
        self.myFont = QtGui.QFont("Arial", self.size);

        self.GLViewWidget.renderText(self.X, self.Y, self.Z, self.text, self.myFont)

class DET_SHOW(object):
    def __init__(self,data):
        self.sipm = data['SIPM']['size']
        self.app  = pg.QtGui.QApplication([])
        self.w    = gl.GLViewWidget()
        self.w.setBackgroundColor([0,0,0])
        self.data = data

    def np2cart(self,p):
        return {'x':p[0],'y':p[1],'z':p[2]}
    def cart2np(self,p):
        return [p['x'],p['y'],p['z']]
    def cyl2np(self,p):
        return [p['r'],p['z'],p['phi']]
    def np2cyl(self,p):
        return {'r':p[0],'z':p[1],'phi':p[2]}

    def cart2cyl(self,p):
        #x=p[0];y=p[1];z=p[2]
        q={}
        q['r'] = np.sqrt(np.square(p['x'])+np.square(p['y']))
        q['z'] = p['z']
        q['phi'] = np.arctan2(p['y'],p['x'])
        return q

    def cyl2cart(self,p):
        q={}
        q['z'] = p['z']
        q['x'] = p['r']*np.cos(p['phi'])
        q['y'] = p['r']*np.sin(p['phi'])
        return q

    def rot_cyl(self,p_array,phi):
        r = np.zeros(p_array.shape)
        for i in range(p_array.shape[0]):
            r[i,:] = self.cyl2np(self.cart2cyl(self.np2cart(p_array[i,:])))
            r[i,2] += phi
            r[i,:] = self.cart2np(self.cyl2cart(self.np2cyl(r[i,:])))
        return r


    def SiPM_QT(self,position,phi,name,photons,max_photons,id=False,
                show_photons=True, MU_LIN=True, TH=0, color2=[0,0,0]):
        s = np.array(self.sipm,dtype=float)
        p = np.array([[s[0]/2,s[1]/2,s[2]/2],
                      [s[0]/2,s[1]/2,-s[2]/2],
                      [-s[0]/2,s[1]/2,s[2]/2],
                      [-s[0]/2,s[1]/2,-s[2]/2],
                      [s[0]/2,-s[1]/2,s[2]/2],
                      [s[0]/2,-s[1]/2,-s[2]/2],
                      [-s[0]/2,-s[1]/2,s[2]/2],
                      [-s[0]/2,-s[1]/2,-s[2]/2]])

        p=self.rot_cyl(p,phi)
        position3d = np.array([position for i in range(p.shape[0])])
        p = p + position3d

        #plt = gl.GLScatterPlotItem(pos=p, color=pg.glColor('w'),size=1)
        MU=25
        if MU_LIN==True:
            color = int(np.log(1+MU*photons/max_photons)/np.log(1+MU)*200.0)
        else:
            color = int(photons/max_photons*200)

        rgb=np.array([0.75*color+55,color+55,0.25*color+55])
        color  = pg.glColor(*rgb)
        color2 = pg.glColor(*(np.array(color2)))

        meshdata = gl.MeshData(vertexes=p,
                               faces=np.array([ [0,1,4], [5,4,1],
                                                [2,0,6], [4,6,0],
                                                [0,2,3], [1,0,3],
                                                [6,2,3], [7,3,6],
                                                [7,3,5], [1,3,5]]),
                          faceColors=np.array([ color,   color,
                                                color,    color,
                                                color,    color,
                                                color2,    color2,
                                                color,    color]))

        plt = gl.GLMeshItem(meshdata=meshdata, color = None,
                            edgeColor=pg.glColor('w'),
                            drawEdges=False,
                            smooth=True,
                            drawFaces=True,
                            shader= 'shaded',#'balloon',#None, #'shaded',
                            glOptions='opaque',
                            computeNormals='False')

        #plt.setGLOptions('additive')
        self.w.addItem(plt)

        if (id==True):
            t = GLTextItem( X=position[0],
                            Y=position[1],
                            Z=position[2],
                            text=str(int(name)),size=4)

            t.setGLViewWidget(self.w)
            self.w.addItem(t)


        text_color = (np.array([rgb[0],255,255])-rgb)*0.75
        text_size = self.w.opts['distance']

        if (show_photons==True and photons>TH) :
            t = GLTextItem( X=position[0],
                            Y=position[1],
                            Z=position[2],
                            text=str(int(photons)),size=8,
                            color=text_color)

            t.setGLViewWidget(self.w)
            self.w.addItem(t)


    def __call__(self,sensors,data,event,ident=False,show_photons=True,
                MU_LIN=True,TH=0):
        items = []

        L1_Slice, SiPM_Matrix_I, SiPM_Matrix_O, topology = DAQ.SiPM_Mapping(self.data, 'striped')

        for i in list(self.w.items):
            self.w.removeItem(i)

        max_light = float(data[event,:].max())
        print max_light
        count=np.zeros(3)+np.array([20,50,50])
        cnt=0

        # for j in SiPM_Matrix_I: #sensors:
        #     for k in j:
        #         k = k+1000 # Paola's style
        #         i = sensors[int(np.argwhere(sensors[:,0]==k))]
        #         self.SiPM_QT(i[1:].transpose(),
        #                      np.arctan2(i[2],i[1]),i[0],
        #                      data[event,int(i[0]-1000)],
        #                      max_light,
        #                      id = ident,
        #                      show_photons=show_photons,
        #                      MU_LIN=MU_LIN,
        #                      TH=TH
        #                      )

        color_map = [[1,0,0],[0,1,0],[0,0,1],
                     [1,1,0],[0,1,1],[1,0,1],
                     [2.5,0,0],[0,2.5,0],[0,0,2.5],
                     [2.5,2.5,0],[0,25.,2.5],[2.5,0,2.5]]
        color_i = 100

        for m in L1_Slice:
            # count = [0,0,0]
            # cnt = np.random.randint(0,2)
            # count[cnt] = np.random.randint(80,255)

            for j in m: #sensors:
                for k in j:
                    k = k+1000 # Paola's style
                    i = sensors[int(np.argwhere(sensors[:,0]==k))]
                    self.SiPM_QT(i[1:].transpose(),
                                 np.arctan2(i[2],i[1]),i[0],
                                 data[event,int(i[0]-1000)],
                                 max_light,
                                 id = ident,
                                 show_photons=show_photons,
                                 MU_LIN=MU_LIN,
                                 TH=TH,
                                 color2=[color_i*color_map[cnt][2],
                                         color_i*color_map[cnt][1],
                                         color_i*color_map[cnt][0]]
                                 )
            if cnt < 11:
                cnt += 1
            else:
                cnt = 0



        t = GLTextItem( X=0, Y=0, Z=0, text=str(event), size=12)
        t.setGLViewWidget(self.w)
        self.w.addItem(t)

        self.w.opts['distance']=500
        self.w.show()

        pg.QtGui.QApplication.exec_()



if __name__ == '__main__':

    path = "/home/viherbos/DAQ_DATA/NEUTRINOS/LESS_4mm/"
    #filename = "daq_output_infinity_4mm_16_2_5_buf800_M"
    jsonfilename = "test_1" #"infinity_4mm_16_2_5_buf800_M"
    filename     =  "p_FR_infinity_4mm_0"#"daq_output_test_1" #"p_FR_infinity_4mm_0"


    positions = np.array(pd.read_hdf(path+filename+".h5",key='sensors'))
    data = np.array(pd.read_hdf(path+filename+".h5",key='MC'), dtype = 'int32')


    SIM_CONT=conf.SIM_DATA(filename=path+jsonfilename+".json",read=True)
    B = DET_SHOW(SIM_CONT.data)


    B( positions, data, event=20,
       ident=False,
       show_photons=True,
       MU_LIN=True,
       TH=1
     )
