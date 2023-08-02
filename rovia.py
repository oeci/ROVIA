# Copyright 2023 Ocean Exploration Cooperative Institute (OECI)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import cv2
import numpy as np
import os
from keras.models import load_model, Model
import keras.backend as K
import tensorflow as tf
from re import finditer, search
import argparse
from moviepy.editor import *
import datetime
from multiprocessing import Pool
import time

WINDOW_SIZE = 60
RESIZE_WIDTH = 160
RESIZE_HEIGHT = 90
NUM_PROCESSES = os.cpu_count()
PROCESS_CHUNK_DURATION = WINDOW_SIZE * 10

class rovia():

    def __init__(self):
        pass

    def recall_m(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision_m(self, y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    def f1_m(self, y_true, y_pred):
        precision = self.precision_m(y_true, y_pred)
        recall = self.recall_m(y_true, y_pred)
        return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

    def readModel(self, path='./grama.hdf5'):
        print('Loading model at: '+path)
        dependencies = {
            'f1_m': f1_m,
            'precision_m': precision_m,
            'recall_m': recall_m
        }
        model = load_model(path, custom_objects=dependencies)
        print('Model load complete ...')
        #model.summary()
        return model

    def generateHighlights(self, X, model, fps):
        y = []
        #Cut up video into bite size chunks
        for i in range(0, len(X), 5*30):
            try :
                X_chunk = X[i:i+5*30]
            except:
                X_chunk = X[i:len(X)]

            prediction = model.predict(X_chunk, verbose=1)
            prediction = np.argmax(prediction, axis=1)
            y.extend(prediction)
        return np.array(y)

    def postProcessHighlights(self, y, kernelSize=5):
        kernel = np.ones(kernelSize) / kernelSize
        data_convolved = np.convolve(y, kernel, mode='same')

        y = 1*[elem>0.5 for elem in data_convolved]
        y_extended = np.array(y)

        for i in range(2, len(y)-2):
            if y[i] == 0 and y[i+1] == 1:
                y_extended[i-2:i+1] = 1
            if y[i-1] == 1 and y[i] == 0:
                y_extended[i:i+3] = 1
        return y_extended
    
    def readVideoChunk(self, path, startFrame, chunkDuration):
        vid = cv2.VideoCapture(path)
        vid.set(cv2.CAP_PROP_POS_FRAMES, startFrame)

        ret, prev_frame = vid.read()
        frame_resized = cv2.resize(prev_frame, (RESIZE_WIDTH, RESIZE_HEIGHT), interpolation=cv2.INTER_AREA)
        prev_gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

        Xmatrix = []
        for frame in range(chunkDuration):
            ret, frame = vid.read()
            if ret == True:
                frame_resized = cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT), interpolation=cv2.INTER_AREA)
                gray = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                # Computes the magnitude and angle of the 2D vectors
                magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                magnitude = magnitude * 255
                magnitude = magnitude.astype('uint8')
                angle = angle / (2 * np.pi) * 255
                angle = angle.astype('uint8')
                frame = np.stack((gray, magnitude, angle), axis=2)
                Xmatrix.append(frame)
                prev_gray = gray
            else:
                break
            
        vid.release()
        cv2.destroyAllWindows()

        windowed = [Xmatrix[i:i + WINDOW_SIZE] for i in range(0, chunkDuration, WINDOW_SIZE)]
        npad = WINDOW_SIZE - np.shape(windowed[-1])[0]
        pad_matrix = np.zeros([npad, RESIZE_HEIGHT, RESIZE_WIDTH, 3], dtype='uint8')
        windowed[-1] = np.append(windowed[-1], pad_matrix, axis=0)
            
        return windowed
    
    def getVideoMetadata(self, path):
        vid = cv2.VideoCapture(path)
        fps = vid.get(cv2.CAP_PROP_FPS)
        frameCount = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        vid.release()
        cv2.destroyAllWindows()

        return fps, frameCount

    def analyzeVideo(self, path, model, verbose):
        fps, frameCount = self.getVideoMetadata(path)

        # Number of video chunks to process
        numChunks = frameCount // PROCESS_CHUNK_DURATION

        results = []
        # for each group of chunks
        for chunkGroupStart in range(0, numChunks, NUM_PROCESSES):
            remainingChunks = numChunks - chunkGroupStart
            numChunksInGroup = min(remainingChunks, NUM_PROCESSES)
            chunkGroupEnd = chunkGroupStart + numChunksInGroup
            chunkGroupStartFrame = chunkGroupStart*PROCESS_CHUNK_DURATION      

            if(verbose):
                print('Analyizing chunks: ' + str(chunkGroupStart) + ' to ' + str(chunkGroupEnd))
                print('Reading chunks ...')

            # Generate the list of video chunk start frames within the chunk group
            pool_args = [(path, chunkGroupStartFrame + (j * PROCESS_CHUNK_DURATION), PROCESS_CHUNK_DURATION) for j in range(numChunksInGroup)]
            
            # split up the work across NUM_PROCESSES processes
            groupResults = []
            with Pool(processes=NUM_PROCESSES) as pool:
                groupResults = pool.starmap(self.readVideoChunk, pool_args)
            del pool_args

            # convert the list of results into a numpy array, remove the num_processes dimension
            groupResults = np.array(groupResults)
            groupResults = groupResults.reshape((-1, 60, 90, 160, 3))

            # generate highlights for the chunk group
            if(verbose):
                print('Generating highlights for chunks ...')
            chunk_highlights = self.generateHighlights(groupResults, model, fps)
            del groupResults

            # refine the highlights for the chunk group
            if(verbose):
                print('Refining highlights for chunks ...')
            refined_highlights = self.postProcessHighlights(chunk_highlights)
            del chunk_highlights

            results.extend(refined_highlights)

            del refined_highlights

            if(verbose):
                print('Completed chunks: ' + str(chunkGroupStart) + ' to ' + str(chunkGroupEnd))
                print('----------------------------------------')
        
        return results

    def interpretInputMetadata(self, videoFileName):
        # Verifies the filename contains a timestamp (yyyymmddThhmmssZ ex.20181214T1401Z)
        # \d{4} - Year - 4 digits (0-9)
        # (0[1-9]|1[0-2]) - Month - 2 digits (0 + 1-9) or (1 + 0-2)
        # (0[1-9]|[1-2]\d|3[0-1]) - Day - 2 digits (0 + 1-9) or (1-2 + 0-9) or (3 + 0-1)
        # T
        # ([0-1]\d|2[0-3]) - Hour - 2 digits (0-1 + 0-9) or (2 + 0-3)
        # [0-5]\d[0-5]\d - Minute - 2 digits (0-5 + 0-9) (0-5 + 0-9)
        # Z
        pattern = r'\d{4}(0[1-9]|1[0-2])(0[1-9]|[1-2]\d|3[0-1])T([0-1]\d|2[0-3])[0-5]\d[0-5]\dZ'

        timestamp = search(pattern, videoFileName)

        if timestamp:
            timestamp = timestamp.group()
        else:
            raise Exception('Invalid filename. Timestamp not found.')

        preformattedFilename = videoFileName.replace(timestamp, '$[timestamp]')
        
        # returns a datetime object
        timestampDT = datetime.datetime.strptime(timestamp, '%Y%m%dT%H%M%SZ')

        return timestampDT, preformattedFilename
    
    # Attempts to fully close a clip, including its reader and audio reader. This fixes FFMPEG errors on windows machines.
    def closeClip(self, clip):
        try:
            clip.reader.close()
            del clip.reader

            if clip.audio is not None:
                clip.audio.reader.close_proc()
                del clip.audio
            
            del clip
        except Exception:
            pass

    def generateClips(self, annotations, path, fps):
        filename = path.split('/')[-1].split('.')[0]
        fullvideo = VideoFileClip(path)

        #Does clips directory exist?
        clipsdir = "./Rovia_Clips/"
        isExist = os.path.exists(clipsdir)
        if not isExist:
            os.makedirs(clipsdir)

        videoDT, preformattedFilename = self.interpretInputMetadata(filename)

        annotations = ''.join([str(1*item) for item in annotations])

        for match in finditer('1+', annotations):
            clipStartTime = int(match.span()[0]*WINDOW_SIZE/fps) # in seconds
            clipEndTime = int(match.span()[1]*WINDOW_SIZE/fps) # in seconds

            clip = fullvideo.subclip(clipStartTime, clipEndTime)

            clipStartDT = videoDT + datetime.timedelta(seconds=clipStartTime)
            clipDate = clipStartDT.strftime('%Y%m%d')
            clipTime = clipStartDT.strftime('%H%M%S')

            alterredFilename = preformattedFilename.replace('$[timestamp]', f'{clipDate}T{clipTime}Z')

            clip.write_videofile(f"{clipsdir}/{alterredFilename}_HL.mp4", temp_audiofile="./temp-audio.m4a", remove_temp=True, audio_codec="aac")

        self.closeClip(fullvideo)

    def startRovia(self, folder, model, verbose):
        dependencies = {
            'f1_m': self.f1_m,
            'precision_m': self.precision_m,
            'recall_m': self.recall_m
        }
        model = load_model(model, custom_objects=dependencies)
        # Go through directory
        FOLDER = folder
        for root, dirs, files in os.walk(FOLDER):
            for file in files:
                if file.endswith('.mp4') or file.endswith('.mov'):
                    # Read video section
                    if verbose == 1:
                        print('\nReading video: ' + file)

                    videofilepath = FOLDER + file

                    start = time.time()
                    prediction = self.analyzeVideo(videofilepath, model, verbose)

                    if verbose == 1:
                        print('Generating clips ...')

                    self.generateClips(annotations=prediction, path=videofilepath, fps=30)
                    end = time.time()

                    print('Time taken: ' + str(end-start) + ' seconds')

                    if verbose == 1:
                        print('Done')
                    
        print('~~Highlight generation complete~~')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='~~ Rovia: The coolest underwater highlight generator ~~\n Incubated @ Ocean Exploration Cooperative Institute', formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-m', '--model', help='Optional: Path to the model file')
    parser.add_argument('-f', '--folder', help='Required:Folder where videos are stored')
    parser.add_argument('-v', '--verbose', help='Optional:Less or more chatter? 0/1')
    args = parser.parse_args()
    if args.folder == None:
        print('File path missing, try python rovia.py -h for help')
        exit()
    if args.model == None:
        model = './grama.hdf5'
    else:
        model = args.model
    if args.verbose == None:
        verbose = 1
    else:
        verbose = args.verbose
    r = rovia()
    r.startRovia(folder=args.folder, model=model, verbose=1)







