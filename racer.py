# Insperation https://www.codingame.com/ide/puzzle/mad-pod-racing

import sys
import os
import math
import copy
from pathlib import Path
import json
import time
import glob
import wave
import contextlib
import random
import numpy as np
import pygame
import pygame as pg
import pygame.gfxdraw
from pygame.locals import *
from inspect import ismethod
from benchmark import *
from tkinter import Tk

debug = 0
if debug > 1:
    import time
    start_time = time.time()
    from inspect import currentframe, getframeinfo

    




def point(x, y):
    return {'x': x, 'y': y}

def vector(d, m):
    return {'d': d, 'm': m}

def deriv(a, b):
    return {'dx': a[0] - b[0], 'dy': a[1] - b[1]}

def angleDelta(a, b, c = 0, l = 0):
    delta = 0
    targetA = a + 180
    targetB = b + 180
    delta = abs(targetA - targetB) % 360
    deltaB = abs(targetA - delta) % 360
    deltaC = abs(targetB - delta) % 360


    if delta > 180:
        delta = 360 - delta

    if deltaB < deltaC or c < 0:
        delta = delta * -1

    if l:
        delta /= l

    return delta

def targetAngleOffset(target, center, offset):
    return (abs(target - center) % 360) - offset

def angle(local, remote, absolute = True, normalize = True):
    der = deriv(remote, local)
    output = math.atan2(der['dy'], der['dx'])
    if normalize:
        output *= (180 / math.pi)
    if absolute:
        if output < 0:
            output += 360
    return output

def angleDiffrence(start, end):
    diff = (start - end + 180) % 360 - 180
    return diff + 360 if diff < -180 else diff

def distance(local, remote):
    der = deriv(remote, local)
    return abs(math.sqrt((der['dx'] ** 2) + der['dy'] ** 2))

def insideCircle(player, checkpoint, radius):
        # Compare radius of circle
        # with distance of its center
        # from given point
        return True if distance(checkpoint, player) <= radius else False

def insideRect(location, rect):
    return True if location[0] > rect[0] and location[1] > rect[1] and location[0] < rect[0]+rect[2] and location[1] < rect[1]+rect[3] else False





# Must be replaced with custom exp function -- codingame doesn't support numpy
def sigmoid(z):
    return 1/(1 + np.exp(-z))

def leakyrelu(A,z):
    if isinstance(z, float):
        if z<0:
            return A*z
        else:
            return z
    else:
        for i, x in enumerate(z):
            if x[0]<0:
                z[i] = A*x[0]
            else:
                z[i] = x[0]

        return z

class Entity(Root):
    parent = None
    def __init__(self, parent, data = {}):
        super(Entity, self).__init__()
        self.parent = parent
        self.data = data
        self.data['previous'] = {}
        self.results = []

    

    def updateVector(self):
        dert = deriv(self.data['location'], self.data['target']['location'])
        dertd = deriv(self.data['target']['location'], self.data['location'])

        if not "lastTarget" in self.data:
            self.data['lastTarget'] = {'dist': 9999, 'angle': 0, 'location': (0, 0)}
        

        dertp = deriv(self.data['lastTarget']['location'], self.data['location'])
        self.data['lastTarget'] = self.data['target']
        self.data['lastTarget']['currentDist'] = math.sqrt((dertp['dx'] ** 2) + abs(dertp['dy'] ** 2))
        
        

        if 'location' in self.data['previous']:
            der = deriv(self.data['previous']['location'], self.data['location'])
        else: 
            der = deriv(self.data['location'], self.data['location'])

        speed =  math.sqrt(abs(der['dx'] ** 2) + abs(der['dy'] ** 2))

        self.data['target']['angle'] = math.atan2(dert['dy'], dert['dx']) * (180 / math.pi)
        self.data['target']['dist'] = math.sqrt((dertd['dx'] ** 2) + abs(dertd['dy'] ** 2))

        if (der['dy'] != 0):
            angle = (math.atan2(der['dy'],  der['dx']) * (180 / math.pi)) 
        else:
            if 'vector' in self.data['previous']:
                angle = self.data['previous']['vector']['d']
            else:
                angle = self.data['angle']

        self.update(vector(angle, speed), 'vector')
        
        

    def setSpeed(self):
        thrust = 100
        angleOffset = self.data['angle'] - self.data['angleOffset'] % 360
        if abs(angleDiffrence(self.data["vector"]["d"], self.data['target']['angle'])) > 45:
            thrust = thrust * (1 - abs(angleDiffrence(self.data["vector"]["d"], self.data['target']['angle'])) / 180)
            thrust = thrust * 1.25
            thrust = min(20, max(100, int(thrust)))
        
        # print(str(int(thrust)), file=sys.stderr, flush=True)

        # distance = self.data['target']['dist'] * 0.2

        if self.data['target']['type'] == 'checkpoint':
            if not ((self.data['ncid'] == 0) and (self.data['lap'] == 3)):

                # print(f'{abs(self.data['target']['dist'] - 600)} < {abs(self.data["vector"]["m"] * math.pi)} and {angleDiffrence(self.data["vector"]["d"], self.data['target']['angle'])} < 10')
                if abs(self.data['target']['dist'] - 600) < abs(self.data["vector"]["m"] * math.pi) and abs(angleDiffrence(self.data["vector"]["d"], self.data['target']['angle'])) < 22:
                    # thrust = thrust * ((1 - abs(self.data['angleOffset'])) / 180)
                    thrust = thrust * (1 - (self.data['target']['dist'] / 2800)) * 2
                    thrust = min(20, max(100, int(thrust)))

                    if self.data['abortEarlyTurn'] == False:
                        # thrust = 0
                        self.data['earlyTurn'] = True
                        

        
        self.update("BOOST" if self.boost() else int(thrust), 'thrust')

    def boost(self):
        if self.parent.data['boost']:
            if self.data['target']['type'] == 'checkpoint':
                if self.parent.data['tick'] >= 25:
                    if self.data['target']['dist'] > 4500:
                        if ((self.data['target']['angle'] <= 1 and self.data['target']['angle'] >=0) or (self.data['target']['angle'] >= -1 and self.data['target']['angle'] <=0)):
                            self.parent.data['boost'] = False
                            return True

        return False

    def setMove(self):
        delta = angleDelta(self.data['target']['angle'], self.data['vector']['d'], targetAngleOffset(self.data['angle'], self.data['target']['angle'], self.data['angleOffset']), 8)

        delta *= math.pi / 180


        x = math.cos(delta) *  (self.data['target']['location'][0] - self.data['location'][0]) - math.sin(delta) * (self.data['target']['location'][1] - self.data['location'][1]) + self.data['location'][0]
        y = math.sin(delta) *  (self.data['target']['location'][0] - self.data['location'][0]) + math.cos(delta) * (self.data['target']['location'][1] - self.data['location'][1]) + self.data['location'][1]

        a = angle(self.data["location"], (x, y), True, True)
        self.results = [a / 360, self.data['thrust'] / 100 if not isinstance(self.data['thrust'], str) else 1, 1 if isinstance(self.data['thrust'], str) else 0, 0]

        
        if x < 0:
            x = 0
        if x > 15999:
            x = 15999
        if y < 0:
            y = 0
        if y > 8999:
            y = 8999
        
        self.update(point(x, y), 'move')

    def setTarget(self, t, i):
        if t == 'checkpoint':
            if self.data['earlyTurn'] == True:
                i += 1
            if i >= len(self.parent.data[t]):
                i = 0
            
            target = {'location': self.parent.data[t][i]}
        else:
            target = self.parent.data[t][i]

        target['type'] = t

        self.data["target"] = target
        self.updateVector()

        if t == 'checkpoint':
            if (self.data['earlyTurn'] == True and self.data['lastTarget']['dist'] <= self.data['lastTarget']['currentDist']):
                self.data['earlyTurn'] = False
                self.data['abortEarlyTurn'] = True
                # print("turnEarly > 2500: " + str(self.data['earlyTurn']))

        if self.parent.data['tick'] > 1:
            if 'angleOffset' not in self.data:
                self.data['angleOffset'] = self.data['angle'] + self.data['target']['angle']
            self.setSpeed()
            self.setMove()
        else :
            self.data['move'] = point(self.parent.data['checkpoint'][1][0], self.parent.data['checkpoint'][1][1])
            self.data['thrust'] = 100      



class Game(Root):
    def __init__(self):
        super(Game, self).__init__()
        self.data['previous'] = {}

    def createEntity(self, title, data):
        self.add(title, Entity(self, data))
    
    def createItem(self, title, data):
        self.add(title, data)


''' 
    Class Genetic
    Responsable for creation and mutation of neural network structures and gentics
'''
class Genetic():
    saveFile = "brains/best"

    def __init__(self, app, inputs, neurons, layers, outputs, load = True, genes = None, structure = None):
            self.id = 0
            self.lastScore = 0
            self.tested = 0
            self.alpha = 0.0001
            self.app = app
            self.inputs = inputs
            self.neurons = neurons
            self.layers = layers
            self.outputs = outputs
            self.previous = {}
            self.genes = None
            self.geneSet = [9, 0, 98, 49, 81, 16, 48, 8, 108, 5, 2, 32, 75, 50, 180, 64, 72, 100, 18, 4, 192, 25, 3, 12, 162, 147, 196, 125, 20, 121, 45, 128, 27, 80, 1, 169, 36, 144]
            self.load = load
            self.structure = []
            self.dnaSegments = {}
            self.dnaSegments["weights"] = (inputs * neurons) + (neurons ** layers-1) + (neurons * outputs)
            self.dnaSegments["biases"] = (neurons * layers) + outputs

            self.optomizationTarget = None
            self.lastLoss = None

            self.saveFile = self.saveFile + f'I{self.inputs}N{self.neurons}L{self.layers}O{self.outputs}'
            
            self.generateGeneSet()

            if not load:
                if (genes == None or not os.path.exists(self.saveFile)) or not self.load :
                    self.genGenes()
                else:
                    self.genes = genes


                if (structure == None or not os.path.exists(self.saveFile)) or not self.load:
                    self.genStructure()
                    self.alter(1, self.structure)
                else:
                    self.loadStructure(structure)
                    self.mutate()
                    self.alter(1, self.structure)
                
            else:
                self.loadStructure()

            self.previous = {"genes": copy.deepcopy(self.genes), "brain": copy.deepcopy(self.structure)}

    def generateGeneSet(self):
        if self.geneSet == None:
            r = set()
            for i in range(20):
                for j in range(6):
                    y = (i ** 2) * j
                    if y < 200:
                        r.add(y)
            geneSet = list(r)
            geneSet = geneSet[0:63]
            random.shuffle(geneSet)
            random.shuffle(geneSet)
            self.geneSet = geneSet


    def process(self, layer, data):
         return np.dot(data, layer["weights"]) + layer["biases"]
    
    def forward(self, data):
        self.inputs = data
        output = None
        self.results = []
        
        for l in self.structure:
            if output is None:
                z = self.process(l, data)
                self.results.append(z)
                output = sigmoid(z)
            else:
                z = self.process(l, output)
                self.results.append(z)
                result = z.tolist().pop()
                output = []
                for r in result:
                    output.append(sigmoid(r))
            
        # print(f'{self.results}')
        return output
        
    

    def backpropagation(self, z, y, i):

        c = []
        print(f'{z=}')
        print(f'{y=}')
        for j in range(len(y)):
            c.append((sigmoid(z[len(z) - 1][0][j]) - y[j]) ** 2)

        cost = sum(c)

        
        results = []
        for l in range(len(z)):
            l = len(z) - (l + 1)
            r = []
            print(f'{len(z)=} {l=}')
            for j in range(len(y)):
                r.append(z[l][0][j] / ((self.alpha * sigmoid(z[l-1][0][j] if l-1 > 0 else i[j])) - sigmoid(z[l-1][0][j] if l-1 > 0 else i[j])) * \
                        ((sigmoid(z[l][0][j] if l > 0 else i[j])) - sigmoid(z[l][0][j])) / (( z[l][0][j]) - z[l][0][j]) * \
                        ((cost) - cost) / ((sigmoid(z[l][0][j])) - sigmoid(z[l][0][j])))
            results.append((sum(r), r))
        print(f'{results=}')

                    
    def getSaveFile(self):
        return f'{self.saveFile}_r{self.app.games}'
    
    def computeLoss(self, target, result):
        target = np.clip(target, 1e-7, 1-1e-7)
        result = np.clip(result, 1e-7, 1-1e-7)
        # result = 0
        # for i, t in enumerate(clippedTarget):
        #     result += math.log(clippedResults[i]) * t
        # result *= -1

        # loss = -np.log(clippedResults, clippedTarget)

        loss = (result[0] - target[0], result[1] - target[1], result[2] - target[2], result[3] - target[3])
        return loss
    
    def checkImproved(self, target, output):
        loss = self.computeLoss(target, output)
        # if (loss[0] < self.lastLoss[0] or loss[1] < self.lastLoss[1] or loss[2] < self.lastLoss[2] or loss[3] < self.lastLoss[3]) and \
        #     (loss[0] <= self.lastLoss[0] and loss[1] <= self.lastLoss[1] and loss[2] <= self.lastLoss[2] and loss[3] <= self.lastLoss[3]):
        print(list(loss))
        if self.lastLoss > abs(sum(enumerate(list(loss)))):
            self.lastLoss = abs(sum(enumerate(list(loss))))
            return True
        return False

    def optomizeZBF(self, input, target, agent):
        self.lastLoss = self.computeLoss(target, agent.brain.results[1])
        direction = True if self.lastLoss[0] > 0 else False
        for l in range(len(agent.brain.structure)):
            l = len(agent.brain.structure) - 1 - l
            for wr in range(len(agent.brain.structure[l]["weights"])):
                for w in range(len(agent.brain.structure[l]["weights"][wr])):
                    w = w if direction else len(agent.brain.structure[l]["weights"][wr]) - 1 - l
                    agent.brain.structure[l]["weights"][wr][w] += 0.0001
                    output = agent.brain.forward(input)
                    if not self.checkImproved(target, output):
                        agent.brain.structure[l]["weights"][wr][w] -= 0.0002
                        output = agent.brain.forward(input)
                        if not self.checkImproved(target, output):
                            agent.brain.structure[l]["weights"][wr][w] += 0.0001
            for b in range(len(agent.brain.structure[l]["biases"][0])):
                b = b if not direction else len(agent.brain.structure[l]["biases"][0]) - 1 - l
                agent.brain.structure[l]["biases"][0][b] += 0.0001
                output = agent.brain.forward(input)
                if not self.checkImproved(target, output):
                    agent.brain.structure[l]["biases"][0][b] -= 0.0002
                    output = agent.brain.forward(input)
                    if not self.checkImproved(target, output):
                        agent.brain.structure[l]["biases"][0][b] += 0.0001

        return self.checkImproved(target, agent.brain.forward(input))


    def genStructure(self):
        self.structure = []
        for l in range(self.layers):
            if l == 0:
                self.structure.append(self.genLayer(self.inputs, self.neurons))
            elif l == self.layers-1:
                self.structure.append(self.genLayer(self.neurons, self.outputs))
            else:
                self.structure.append(self.genLayer(self.neurons, self.neurons))

    def genGenes(self):
        genes = ""
        for i in range(((self.dnaSegments["weights"] + self.dnaSegments["biases"]) * 7) + 72):
            genes += str(int(bool(random.getrandbits(1))))

        self.genes = genes


    def loadStructure(self, structure = None, file = None):
        if structure is None and os.path.exists(self.saveFile):
            self.structure = []
            # if file == None:
            f = open(self.saveFile, "r")
            # else:
            #     if os.path.exists(file):
            #         print('Loaded Manual File....')
            #         f = open(file, "r")

            data = json.loads(f.read())
            f.close
            
            for e in data:
                if "weights" in e:
                    self.structure.append({"weights": np.array(e["weights"]), "biases": np.array(e["biases"])})
                if "genes" in e:
                    self.genes = e["genes"]

        elif structure is None:
            self.genStructure()
            self.genGenes()
        else:
            self.structure = structure


    def saveStructure(self, maunual = None, score = ''):
        if maunual:
            f = open(maunual, "w")
            f.write(str(self))
            f.close()
        f = open(self.getSaveFile()+str(score), "w")
        f.write(str(self))
        f.close()
        f = open(self.saveFile, "w")
        f.write(str(self))
        f.close()

    def genLayer(self, inputs, outputs):
        weights = np.random.randn(inputs, outputs) * 0.05
        biases = np.zeros((1, outputs))

        return {"weights": weights, "biases": biases}
    
    def __str__(self):
        output = []
        for s in self.structure:
            weights = []
            for w in s["weights"].tolist():
                weights.append(w)
            output.append({"weights": weights, "biases": s["biases"].tolist()})
        output.append({"genes": self.genes})

        return json.dumps(output, indent=3)
    
    def replace(self, type = None, score = 0, generation = 0, partnerGenes = None, partnerBrain = None):

        if score > self.lastScore:
            self.previous = {"genes": copy.deepcopy(self.genes), "brain": copy.deepcopy(self.structure)}
        else:
            self.structure = copy.deepcopy(self.previous["brain"])

        if score >= self.app.scoreboard[9]["highScore"]:
            self.lastScore = score
            return

        if type == "leader":
            self.replaceLeader(score, generation)
        
        if type == "runt":
            self.replaceRunt(score, generation)
        
        if type == "pack":
            self.replacePack(score, generation, partnerGenes, partnerBrain)

        if type == "clone":
            self.replaceClone(score, generation, partnerGenes, partnerBrain)

        self.lastScore = score
   
    
    def replaceLeader(self, score, generation):
        if score == 0:
            self.mutate()
        elif score < self.lastScore and self.tested:
            self.mutate()
        self.alter(generation, self.previous["brain"] if score < self.lastScore else self.structure)
    
    def replaceRunt(self, score, generation):
        self.genStructure()
        if score < self.lastScore and self.tested:
            self.mutate()
        self.alter(generation, self.structure)

    def replacePack(self, score, generation, partnerGenes, partnerBrain):
        if partnerBrain is None:
            partnerBrain = self.structure
        if partnerGenes is None:
            partnerGenes = self.genes

        if score ==  0:
            self.tested = 0
            self.breed(partnerGenes)
            newBrain = []
            for l in range(len(partnerBrain)):
                newBrain.append({"weights": [], "biases": []})
                for wr in range(len(partnerBrain[l]["weights"])):
                    newBrain[l]["weights"].append([])
                    for w in range(len(partnerBrain[l]["weights"][wr])):
                        newBrain[l]["weights"][wr].append(partnerBrain[l]["weights"][wr][w] if bool(random.getrandbits(1)) else self.structure[l]["weights"][wr][w])
                    newBrain[l]["weights"][wr] = np.array(newBrain[l]["weights"][wr])
                newBrain[l]["weights"] = np.array(newBrain[l]["weights"])
                newBrain[l]["biases"].append([])
                for b in range(len(partnerBrain[l]["biases"][0])):
                    newBrain[l]["biases"][0].append(partnerBrain[l]["biases"][0][b] if bool(random.getrandbits(1)) else self.structure[l]["biases"][0][b])
                newBrain[l]["biases"] = np.array(newBrain[l]["biases"])
            self.structure = newBrain
        else:
            if self.tested:
                if score <= self.lastScore:
                    self.tested = 0
                    self.mutate()
            else:
                if score <= self.lastScore:
                    self.tested = 1
                
        self.alter(generation, self.structure)
    
  
                

    def replaceClone(self, score, generation, partnerGenes, partnerBrain):
        if score == 0:
            self.genes = partnerGenes
            self.structure = partnerBrain
            self.mutate()
            self.alter(generation, self.structure)
        elif score < self.lastScore and self.tested:
            self.mutate()
            self.alter(generation, self.structure)
        else:
            self.alter(generation, self.structure)

        

    def mutate(self):
        mutations = max(1, random.randint(0, min(self.app.maxMutations, int(len(self.genes) * 0.01))))
        position = 0
        genes = list(self.genes)
        
        for m in range(mutations):
            location = random.randint(0, len(self.genes)-1)
            if position + location >= len(self.genes) - 1:
                position = (location + position) - position
            else:
                position += location

            genes[position] = str(int(random.getrandbits(1)))

        self.genes = ''.join(genes)
            

    def breed(self, genes):
        if genes is None:
            genes = self.genes
        child = ""
        for gene in range(int(len(self.genes) / 2)):
            if bool(random.getrandbits(1)):
                child += self.genes[gene*2:(gene*2)+2]
            else:
                child += genes[gene*2:(gene*2)+2]
        self.genes = child

    def mate(self, partner):
        children = {1 : {'genes': '', 'brain': []}, 2: {'genes': '', 'brain': []}}
        for gene in range(int(len(self.genes) / 2)):
            if bool(random.getrandbits(1)):
                children[1]['genes'] += self.genes[gene*2:(gene*2)+2]
                children[2]['genes'] += partner.agent.genes[gene*2:(gene*2)+2]
            else:
                children[1]['genes'] += partner.agent.genes[gene*2:(gene*2)+2]
                children[2]['genes'] += self.genes[gene*2:(gene*2)+2]


        for l in range(len(self.structure)):
            children[1]['brain'].append({"weights": [], "biases": []})
            children[2]['brain'].append({"weights": [], "biases": []})
            for wr in range(len(self.structure[l]["weights"])):
                children[1]['brain'][l]["weights"].append([])
                children[2]['brain'][l]["weights"].append([])
                for w in range(len(self.structure[l]["weights"][wr])):
                    if bool(random.getrandbits(1)):
                        children[1]['brain'][l]["weights"][wr].append(partner.agent.genetics.structure[l]["weights"][wr][w])
                        children[2]['brain'][l]["weights"][wr].append(self.structure[l]["weights"][wr][w])
                    else:
                        children[1]['brain'][l]["weights"][wr].append(self.structure[l]["weights"][wr][w])
                        children[2]['brain'][l]["weights"][wr].append(partner.agent.genetics.structure[l]["weights"][wr][w])
                      
                children[1]['brain'][l]["weights"][wr] = np.array(children[1]['brain'][l]["weights"][wr])
                children[2]['brain'][l]["weights"][wr] = np.array(children[2]['brain'][l]["weights"][wr])

            children[1]['brain'][l]["weights"] = np.array(children[1]['brain'][l]["weights"])
            children[2]['brain'][l]["weights"] = np.array(children[2]['brain'][l]["weights"])
            children[1]['brain'][l]["biases"].append([])
            children[2]['brain'][l]["biases"].append([])

            for b in range(len(self.structure[l]["biases"][0])):
                if bool(random.getrandbits(1)):
                    children[1]['brain'][l]["biases"][0].append(partner.agent.genetics.structure[l]["biases"][0][b])
                    children[2]['brain'][l]["biases"][0].append(self.structure[l]["biases"][0][b])
                    
                else:
                    children[1]['brain'][l]["biases"][0].append(self.structure[l]["biases"][0][b])
                    children[2]['brain'][l]["biases"][0].append(partner.agent.genetics.structure[l]["biases"][0][b])

            children[1]['brain'][l]["biases"] = np.array(children[1]['brain'][l]["biases"])
            children[2]['brain'][l]["biases"] = np.array(children[2]['brain'][l]["biases"])

        return children

    def alter(self, generation, structure):
        g = 0
        structure = copy.deepcopy(structure)
        for l in structure:
            for wr in range(len(l["weights"])):
                for w in range(len(l["weights"][wr])):
                    gene = self.genes[g*7:(g*7)+7]
                    change = int(gene[1:], 2) * ((self.app.geneImpact * max(0.0001, ((1 / self.app.totalGenerations) / 4))))
                    if int(gene[0]):
                        change = change * -1
                        
                    
                    # l["weights"][wr][w] += change * -1 if self.tested else 1 if self.app.tick == 1 else change
                    l["weights"][wr][w] += change
                    l["weights"][wr][w] = max(-2, min(2, l["weights"][wr][w]))
                    g += 1

            for b in range(len(l["biases"][0])):
                gene = self.genes[g*7:(g*7)+7]
                change = int(gene[1:], 2) * ((self.app.geneImpact * max(0.0001, ((1 / self.app.totalGenerations) / 4))))
                if int(gene[0]):
                    change = change * -1


                # l["biases"][0][b] += change * -1 if self.tested else 1 if self.app.tick == 1 else change
                l["biases"][0][b] += change
                l["biases"][0][b] = max(-1 * len(l["weights"][0]), min(len(l["weights"][0]), l["biases"][0][b]))

                g += 1

        self.structure = structure

class Player:
    def __init__(self, id, x, y, app, field, shape = ((0, -150), (-173, 150), (0, 50), (173, 150)), wings = ((0, 150), (173, -50), (-173, -50))):
        self.lap = 1
        self.points = ()
        self.angle = 0
        self.speed = 0
        self.speedVector = (0, 0)
        self.agent = None
        self.id = id
        self.app = app
        self.field = field
        # self.location = (x + random.randint(-600, 600), y + random.randint(-600, 600))
        self.location = (x, y)
        self.target = (x, y)
        self.center = pygame.math.Vector2(self.location)
        self.shape = shape
        self.underShape = wings
        self.frames = []
        self.points = self.definePoints(self.shape, self.center)
        self.underPoints = self.definePoints(self.underShape, self.center)
        self.lives = 0
        self.onTop = 0

        self.game = 0
        self.moves = 0
        self.scores = {"distance": 0, "accrued": 0}
        self.totalScore = 0
        self.rank = len(self.app.players)
        self.rounds = 0
        
        self.fieldTransform()
        self.angle = 0
        self.rotate(90)
        self.checkpoint = 1
        self.lastCheckpoint = 1
        self.highestCheckpoint = 0
        self.highScore = 0
        self.lastHighScore = 0
        self.movesSince = 0
        self.badMoves = 0
        self.lastDist = 0
        self.highestLap = 0
        self.brainBackup = None


        self.generations = 1
        self.totalGenerations = 1

        self.angle = angle(self.location, self.app.targets[self.checkpoint].center)

        self.alive = True

        

        self.mapDiag = distance((1200,1200), (self.app.field.width - 1200, self.app.field.height - 1200))

        self.targeted = False

    def reinit(self):
        self.checkpoint = 1
        self.lastCheckpoint = 1

        x = self.app.targets[0].center[0]
        y = self.app.targets[0].center[1]
        self.lap = 1
        self.angle = 0
        self.speed = 0
        self.speedVector = (0, 0)
        # self.location = (x + random.randint(-600, 600), y + random.randint(-600, 600))
        self.location = (x, y)
        self.target = (x, y)
        self.center = pygame.math.Vector2(self.location)
        self.frames = []
        self.points = self.definePoints(self.shape, self.center)
        self.underPoits = self.definePoints(self.underShape, self.center)

        self.game = self.app.games
        self.moves = 0
        self.scores = {"distance": 0, "accrued": 0}
        self.totalScore = 0
        self.lastHighScore = 0
        self.movesSince = 0
        self.badMoves = 0
        self.lastDist = 0
        self.highestCheckpoint = 0

        
        self.fieldTransform()
        self.angle = 0
        self.rotate(90)
        self.angle = angle(self.location, self.app.targets[self.checkpoint].center)

        self.alive = True

        
        checkpoints = []
        for i in range(len(self.app.targets)):
            c_x, c_y = [int(j) for j in self.app.targets[i].center]
            cp = point(c_x, c_y)
            checkpoints.append(cp)

        

    def housekeeping(self):
        
        if ((self.moves > self.app.maxMoves or self.lap > self.app.laps) or (self.badMoves >= 2)):
            self.lives += 1
            self.alive = False
            self.updateScore()
                

                
                
            

    def definePoints(self, shape, center, scale = (1, 1)):
        points = []
        for segment in shape:
            x = segment[0] * scale[0]
            y = segment[1] * scale[1]
            points.append(tuple((center[0] + x, center[1] + y)))
        return tuple(points)

    def rotate(self, angle):
        pp = self.renderCenter
        rotated_points = [
            (pygame.math.Vector2(x, y) - pp).rotate((angle + 90) % 360) + pp for x, y in self.renderPoints]
        self.renderPoints = tuple(rotated_points)
        rotated_points = [
            (pygame.math.Vector2(x, y) - pp).rotate((angle + 90) % 360) + pp for x, y in self.renderPointsUnder]
        self.renderPointsUnder = tuple(rotated_points)
        rotated_points = [
            (pygame.math.Vector2(x, y) - pp).rotate((angle + 90) % 360) + pp for x, y in self.renderUnderPoints]
        self.renderUnderPoints = tuple(rotated_points)
        rotated_points = [
            (pygame.math.Vector2(x, y) - pp).rotate((angle + 90) % 360) + pp for x, y in self.renderUnderPointsUnder]
        self.renderUnderPointsUnder = tuple(rotated_points)

    def fieldTransform(self):
        transform = {"x": self.center[0], "y": self.center[1]}
        transform["x"] *= self.field.width / self.field.grid.width
        transform["x"] += self.field.left
        transform["y"] *= self.field.height / self.field.grid.height
        transform["y"] += self.field.top

        self.renderCenter = pygame.math.Vector2(transform["x"], transform["y"])
        self.renderPoints = self.definePoints(self.shape, self.renderCenter, (self.field.gr * 2))
        self.renderPointsUnder = self.definePoints(self.shape, self.renderCenter, (self.field.gr * 2))
        self.renderUnderPoints = self.definePoints(self.underShape, self.renderCenter, (self.field.gr * 2))
        self.renderUnderPointsUnder = self.definePoints(self.underShape, self.renderCenter, (self.field.gr * 2))

        transform = {"x": self.target[0], "y": self.target[1]}
        transform["x"] *= self.field.width / self.field.grid.width
        transform["x"] += self.field.left
        transform["y"] *= self.field.height / self.field.grid.height
        transform["y"] += self.field.top

        self.renderTarget = pygame.math.Vector2(transform["x"], transform["y"])

    def move(self, x = 0, y = 0, thrust = 100):
        self.target = (x, y)
        self.generateFrames(thrust)
        self.agent.update(self.target, "location")


    def generateFrames(self, thrust):
        if not self.alive:
            return
        
        self.frames = []
        if not isinstance(thrust, int):
            if thrust == "BOOST":
                thrust = 650
            elif thrust == "SHIELD":
                thrust = 0
            elif int(thrust) > 0:
                thrust = int(thrust) / self.app.frameDivisor

        for i in range(self.app.frameDivisor):
            self.faceVector =  thrust * (math.cos(math.pi * 2 * self.angle / 360)), thrust * (math.sin(math.pi * 2 * self.angle / 360))
            self.speedVector = tuple([x + y for x, y in zip(self.faceVector, self.speedVector)])
            self.center = pygame.math.Vector2(tuple([x + y for x, y in zip(self.center, self.speedVector)]))

            self.updateDistance()
            self.updateAngle()
            
            
            self.location = self.center
            self.speedVector = tuple([int((cord * 0.85)/self.app.frameDivisor) for cord in self.speedVector])
            
            if self.rank < 10 or self.app.hidden == False:
                self.fieldTransform()
                self.rotate(self.angle)
                self.frames.append({"center": self.renderCenter, "points": self.renderPoints, "underPoints": self.renderUnderPoints, "underPointsUnder": self.renderUnderPointsUnder, "pointsUnder": self.renderPointsUnder})
                
        if len(self.frames):
            self.frames = list(reversed(self.frames))

    def updateAngle(self):
        # Rotate towards target at a max rate of 18 degress per tick
        diff = angleDiffrence(self.angle, angle(self.center, self.target, True))
        if diff < 0:
            angleChange = (self.angle + (18/self.app.frameDivisor if abs(diff) > 18/self.app.frameDivisor else abs(diff)/self.app.frameDivisor)) % 360
        else:
            # self.angle = (self.angle - (18 if abs(diff) > 18 else abs(diff))) % 360h
            angleChange = (self.angle - (18/self.app.frameDivisor if abs(diff) > 18/self.app.frameDivisor else abs(diff)/self.app.frameDivisor)) % 360

        if abs(diff) > 2:
            self.angle = angleChange

            # self.angle = self.angle if self.angle < 0 else 360 - self.angle if self.angle < 360 else self.angle % 360

    def updateDistance(self):
        target = self.app.targets[self.checkpoint]
        myDistance = distance(self.location, target.center)

        range = min(1, (1 - ((myDistance - target.size) / (target.dist - target.size))))
        score = math.ceil(((self.app.totalTargetDist / len(self.app.targets)) * range))
        if score < 0:
            if self.app.trainingStage == 0:
                score = int(score / 1024)
            if self.app.trainingStage == 1:
                score = int(score / 8)
            if self.app.trainingStage == 2:
                score = int(score * 2)
                score = score + math.ceil(score * (1 - (self.moves / self.app.maxMoves)))
        if (score < 0 and self.app.trainingStage > 0) or score > self.scores["distance"]:
            self.scores["distance"] = score
        self.updateScore()
        # if self.lap > 1:
        #     self.scores["distance"] = int((((1 - (self.moves / self.app.maxMoves) * self.scores["distance"]))))


        if myDistance <= self.app.targets[self.checkpoint].size:
            self.lastCheckpoint = self.checkpoint
            self.checkpoint = ((self.checkpoint + 1) % (len(self.app.targets)))
            self.insideCheckpoint()

        if self.checkpoint > self.highestCheckpoint:
            self.highestCheckpoint = self.lastCheckpoint

        if self.app.highestCheckpoint < self.lastCheckpoint:
            self.app.highestCheckpoint = self.lastCheckpoint

        if abs(myDistance) >= abs(self.lastDist):
            self.badMoves += 1
        else:
            if self.badMoves > 0:
                self.badMoves = 0

        self.lastDist = myDistance
        
    def insideCheckpoint(self):
            self.badMoves = -12
            self.scores["accrued"] += int(self.scores["distance"]) + 5000
            self.scores["distance"] = 0
            self.updateScore()
            self.moves = 0

            topId = self.app.scoreboard[0]["id"]

            if self.onTop >= 3000:
                self.app.games += 1
                self.app.reinit()
                return

            if self.lastCheckpoint == 0:
                if self.app.trainingStage < 1:
                    self.app.trainingStage = 1
                self.lap += 1
                if self.lap > self.highestLap:
                    self.highestLap = self.lap
                    if self.highestLap > self.app.highestLap:
                        self.app.highestLat = self.highestLap
                if self.lap > self.app.laps:
                    if self.app.trainingStage < 2:
                        self.app.trainingStage = 2
                    self.alive = False
                    for player in self.app.players:
                        player.alive = False
                    return 
                    if self.id == topId and hasattr(self.app.audio, 'sound_finish'):
                        if not self.app.audio.muted:
                            self.app.audio.sound_lap.set_volume(0.1)
                            self.app.audio.sound_lap.play()
                    self.rounds += 1
                    self.app.rounds = self.rounds
                    
                    if self.rounds >= 3 : # and self.id != len(self.app.scoreboard) -1
                        self.app.games += 1
                        self.app.reinit()

                    self.reinit()
                    return
                if self.id == topId and hasattr(self.app.audio, 'sound_lap'):
                    if not self.app.audio.muted:
                        self.app.audio.sound_lap.set_volume(0.1)
                        self.app.audio.sound_lap.play()
            if self.id == topId and hasattr(self.app.audio, 'sound_checkpoint'):
                if not self.app.audio.muted:
                    self.app.audio.sound_checkpoint.set_volume(0.1)
                    self.app.audio.sound_checkpoint.play()

    def updateScore(self):
        self.moves += 1
        self.movesSince += 1
        self.lastHighScore = self.highScore
        self.totalScore = sum(self.scores.values())
        self.movesSince = 0


class Interface:
    called = False
    times = 0
    sound_checkpoint = None
    def __init__(self, app):
        self.app = app

        # Called once
    def inital(self):
        # Laps
        if self.times == 0:
            self.times += 1
            return str(self.app.laps)
        # Checkpoints
        if self.times == 1:
            self.times += 1
            return str(len(self.app.targets))
        # Each Checkpoint[Checkpoints]
        if self.times >= 2:
            if self.times - 1 <= len(self.app.targets):
                output = f'{int(self.app.targets[self.times - 2].center[0])} {int(self.app.targets[self.times - 2].center[1])}'
                self.times += 1
                if self.times - 2 == len(self.app.targets):
                    self.times = 0
                    self.called = True
                return output

        # Called every frame
    def input(self):
        # print(self.called)
        if not self.called:
            return self.inital()
        # Each (Opponet)Pods[pod]
        # x, y
        # speed vector x, y
        # Angle
        # Next Checkpoint
        if self.times < len(self.app.players):
            p = self.app.players[self.times]
            self.times += 1
            if self.times >= len(self.app.players):
                # print(f'Made it here!?!?')
                self.times = 0
            return f'{int(p.center[0])} {int(p.center[1])} {int(p.speedVector[0])} {int(p.speedVector[1])} {int(p.angle)} {int(p.checkpoint if p.checkpoint < len(self.app.targets) else 0 )}'

        # Process Output string (X, y, (0-100 | BOOST | SHIELD))
        # Called once per pod
    def command(self, command):
        if self.times < len(self.app.players):
            x, y, t = command.split()
            if self.app.players[self.times].id == self.app.scoreboard[0]["id"] and hasattr(self.app, 'app.sound_ship'):
                if not self.app.audio.muted:
                    v = int(t) if len(t) <= 3 else 200 if t == "BOOST" else 10
                    self.app.sound_ship.set_volume((v  * 0.001) / 2)
            self.app.players[self.times].move(int(x), int(y), int(t) if len(t) <= 3 else t)
            self.times += 1
            if self.times >= len(self.app.players):
                self.times = 0

class Field:
    width = 800
    height = 600
    border = 5
    top = 0

    def __init__(self, app, window, border = 5, top = 0, left = -1, width = 800, height = 600):
        self.app = app
        self.window = window
        self.top = top
        self.width = width
        self.height = math.ceil(width * 0.5625)
        self.left = left
        self.border = border
        self.path = 'assets/img/'
        self.image = None
        self.cloudMask = []
        self.cmidx = 0
        self.cloudRate = 2

        if self.left == -1:
            self.left = window.width - self.width

        self.grid = Grid(self)
        self.fieldRect = pygame.Rect(self.left, self.top, self.width, self.height)
        self.borderRect = pygame.Rect(self.left - self.border, self.border * -1, self.width + (self.border * 2), self.height + (self.border * 2))
        self.gr = (self.width / self.grid.width, self.height / self.grid.height)

    def updateSize(self):
        self.width = self.window.width - 224
        self.height = math.ceil(self.width * 0.5625)
        self.left = self.window.width - self.width
        self.fieldRect = pygame.Rect(self.left, self.top, self.width, self.height)
        self.borderRect = pygame.Rect(self.left - self.border, self.border * -1, self.width + (self.border * 2), self.height + (self.border * 2))
        self.gr = (self.width / self.grid.width, self.height / self.grid.height)

    def loadImages(self):
        self.loadBackground()
        self.loadClouds()

    def loadBackground(self):
        p = Path(self.path)
        f = p / 'background.png'

        if f.exists():
            self.image = pg.Surface.convert_alpha(pg.image.load(f))
            return True

        return False
    
    def animate(self):
        if len(self.cloudMask):
            if not self.app.tick % int(self.cloudRate / 2):
                for mask in self.cloudMask:
                    # maskRect = mask.get_rect()
                    # rightEdge = maskRect.clip(maskRect.width - 2, 0, 2, maskRect.height)
                    # bottomEdge = maskRect.clip(0, maskRect.height - 3, maskRect.width - 2, 3)
                    # main = maskRect.clip(0, 0, maskRect.width - 2, maskRect.height - 3)
                    # mask.fill((255, 255, 255, int(255 * 0)))
                    # mask.blit(mask, main)
                    mask.scroll(2, 3)
            
            if not self.app.tick % self.cloudRate:
                self.cmidx += 1
                self.cmidx = self.cmidx % len(self.cloudMask)
                
            self.app._display_surf.blit(self.cloudMask[self.cmidx], (self.fieldRect[0], 1 - (self.image.get_height() - self.fieldRect[3])))
    
    def loadClouds(self):
        p = Path(self.path)
        f = p / 'cloudMask_*.png'

        clouds = glob.glob(self.path + 'cloudMask_*.png')

        if len(clouds):
            self.cloudMask = []
            for cloud in clouds:
                cloudImg = pg.Surface.convert_alpha(pg.image.load(cloud))
                cloudImg.set_alpha(64)
                self.cloudMask.append(cloudImg)

            return True
        return False


    def __str__(self):
        return {"field": self.fieldRect, "border": self.borderRect}
    
class Target:

    def __init__(self, field, x, y, color = None, size = 600):
        self.center = (x, y)
        colors = Colors()
        self.color = colors.red if color == None else color
        self.size = size 
        self.field = field
        self.fieldTransform()

    def fieldTransform(self):
        transform = {"x": self.center[0], "y": self.center[1]}
        transform["x"] *= self.field.width / self.field.grid.width
        transform["x"] += self.field.left
        transform["y"] *= self.field.height / self.field.grid.height
        transform["y"] += self.field.top

        self.renderCenter = pygame.math.Vector2(transform["x"], transform["y"])

class Grid:
    width = 16000
    height = 9000

    def __init__(self, field):
        self.field = field
        top = 0
        left = 0

class Window:
    width = 1024
    height = 768
    # width = 1920
    # height = 1400
    def __init__(self):
        pass

    def size(self):
        return (self.width, self.height)
    
    def updateSize(self, w, h):
        self.width = w
        self.height = h

class Colors:
    black = (0, 0, 0)
    blue = (20, 30, 200)
    cyan = (20, 200, 200)
    yellow = (200, 200, 20)
    white = (255, 255, 255)
    grass = (5, 40, 25)
    green = (20, 200, 30)
    magenta = (200, 20, 200)
    purple = (160, 32, 240)
    pink = (197, 58, 127)
    dark_green = (0, 10, 6)
    dark_gray = (12, 12, 12)
    orange = (255, 165, 0)
    light_blue = (25, 60, 255)
    red = (200, 20, 30)

    tblack = (0, 0, 0, 191)
    tclear = (0, 0, 0, 0)
    tgreen = (20, 200, 30, 127)
    twhite = (255, 255, 255, 127)
    tgrass = (5, 40, 25, 64)
    tdark_green = (0, 10, 6, 127)
    tdark_gray = (12, 12, 12, 127)
    torange = (255, 165, 0, 64)
    tlight_blue = (25, 60, 255, 127)
    tred = (200, 20, 30, 127)

    targetColors = [blue, cyan, yellow, green, magenta, purple, pink, red]

class Audio:
    pwd = f"{os.path.dirname(__file__)}\\audio"
    tracks = []
    sounds = []
    activeSounds = []
    start = None
    end = 0
    muted = False
    def __init__(self):
        pygame.mixer.init()
        pygame.mixer.music.set_volume(0)

        tracks = glob.glob("./assets/audio/tracks/*.wav")
        for track in tracks:
            with contextlib.closing(wave.open(track,'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / float(rate)
                loops = 1 if duration > 120 else math.ceil(120 / duration)
                self.tracks.append({"file": track, "name": track.split('\\')[1], "duration": duration, "loops": loops})

        sounds = glob.glob("./assets/audio/effects/*.wav")
        for sound in sounds:
            with contextlib.closing(wave.open(sound,'r')) as f:
                frames = f.getnframes()
                rate = f.getframerate()
                duration = frames / float(rate)
                loops = 1 if duration > 120 else math.ceil(120 / duration)
                self.sounds.append({"file": sound, "name": sound.split('\\')[1],"duration": duration, "loops": loops})

    def load(self):
        soundFile = next((item for item in self.sounds if item["name"] == "checkpoint.wav"), None)
        if soundFile != None:
            self.sound_checkpoint = pygame.mixer.Sound(soundFile["file"])
            self.activeSounds.append(self.sound_checkpoint)
            self.sound_checkpoint.set_volume(0.1)

        soundFile = next((item for item in self.sounds if item["name"] == "lap.wav"), None)
        if soundFile != None:
            self.sound_lap = pygame.mixer.Sound(soundFile["file"])
            self.activeSounds.append(self.sound_lap)
            self.sound_lap.set_volume(0.1)

        soundFile = next((item for item in self.sounds if item["name"] == "finish.wav"), None)
        if soundFile != None:
            self.sound_finish = pygame.mixer.Sound(soundFile["file"])
            self.activeSounds.append(self.sound_finish)
            self.sound_finish.set_volume(0.2)

        soundFile = next((item for item in self.sounds if item["name"] == "ship.wav"), None)
        if soundFile != None:
            self.sound_ship = pygame.mixer.Sound(soundFile["file"])
            self.activeSounds.append(self.sound_ship)
            self.sound_ship.set_volume(0.001)
            self.sound_ship.play(-1)                

    def play(self):
        if len(self.tracks) and not self.muted:
            if self.fade() > 0:
                return
            pygame.mixer.music.set_volume(0.1)
            trackNumber = random.randint(0, len(self.tracks)-1)
            pygame.mixer.music.load(self.tracks[trackNumber]["file"])
            self.start = time.time()
            self.end = self.start + ((self.tracks[trackNumber]["duration"] * self.tracks[trackNumber]["loops"]) - 5)
            pygame.mixer.music.play(self.tracks[trackNumber]["loops"]-1)
    
    def fade(self):
        now = time.time() 
        if now >= self.end-8:
            pygame.mixer.music.fadeout(5000)
        if now >= self.end:
            pygame.mixer.music.set_volume(0)
        return pygame.mixer.music.get_volume()
    
    def mute(self):
        self.muted = not self.muted
        if self.mute:
            volume = 0
        else:
            volume = 0.1

        pygame.mixer.music.set_volume(volume)
        for sound in self.activeSounds:
            sound.set_volume(volume)
    

class Inputs:
    def __init__(self, app):
        self.app = app
        inputs = [{"variable": self.app.maxMoves, "input": int, "string": "Max Moves: {name:>3}"},\
                  {}]
        self.vaiables = ["maxMoves", "maxLives", "geneImpactDefault", "maxMutations", "leaderPct",\
                         "runtPct", "clonePct", "frameDivisor", "fps", "hidden", "debug", "reinit"]
        self.strings = ["Max Moves: {name:>3}",\
                   "Max Lives: {name:>3}",\
                   "Gene Impact: {name:>3}",\
                   "Max Mutations: {name:>3}",\
                   "Leader Pct: {name:>3}",\
                   "Runt Pct: {name:>3}",\
                   "clone Pct: {name:>3}",\
                   "Frame Divisor: {name:>3}",\
                   "FPS: {name:>3}",\
                   "Hidden: {bool(name)}",\
                   "Debug: {bool(name)}",\
                   "New Level"]
        self.elements = []
        self.active = None
        self.entry = None

        self.defineInputs()

    def save(self, idx):
        if "." in self.entry:
            val = float(self.entry)
        else:
            val = int(self.entry)

        setattr(self.app, self.vaiables[idx], val)
        self.entry = None
    
    def load(self, idx):
        if not callable(getattr(self.app, self.vaiables[idx], None)):
            self.entry = str(getattr(self.app, self.vaiables[idx]))

    def fstr(self, template, name):
        if hasattr(self.app, name):
            name = getattr(self.app, name)

        return eval(f'f"""{template}"""')

    def defineInputs(self):
        areaLeft = self.app.field.left + 3
        areaTop = self.app.field.height + 5
        paddingRight = 4
        paddingBottom = 4
        self.elements = []

        top = areaTop
        left = areaLeft
        for i, s in enumerate(self.strings):
            name = self.vaiables[i]
            s = self.fstr(s, name)
            
            self.elements.append({"idx": i, "size": self.generateSize(s)})

            if len(self.elements) > 1:
                top = self.elements[i-1]["position"][3]
                if self.elements[i-1]["position"][3] + self.elements[i]["size"][1] > self.app.window.height:
                    self.elements[i]["position"] = (1, 1, 1, 1)
                    left = self.calculateNextColumn()
                    top = areaTop

                self.elements[i]["position"] = (left, top, left+self.elements[i]["size"][0]+paddingRight, top+self.elements[i]["size"][1]+paddingBottom)
            else:
                self.elements[i]["position"] = (left, top, left+self.elements[i]["size"][0]+paddingRight, top+self.elements[i]["size"][1]+paddingBottom)

            self.elements[i]["rect"] = (left - 1, top - 1, self.elements[i]["size"][0] + 2, self.elements[i]["size"][1] + 2)


    def calculateNextColumn(self):
        longest = sorted(self.elements, key=lambda r: r["position"][2], reverse=False).pop()
        return longest["position"][2]


    def generateSize(self, string):
        t = self.app.font.render(string, True, (0, 0 ,0), (0, 0 ,0))
        return self.app.font.size(string)
         
        
class App:

    frameDivisor = 1 # 6 is also a pratical value...
    framesRemaining = 0
    maxLives = 9
    maxMoves = 45
    geneImpactDefault = 0.1
    geneImpact = 0.1
    maxMutations = 8
    matingStrat = "MateUp" # "Populations"

    leaderPct = 0.1
    runtPct = 0.2
    clonePct = 0.5
    
    targetProfiles = [((3185, 2319), (8272, 1761), (8506, 5423), (14141, 7696), (1809, 6918)),
                      ((1200, 9000/2), (((16000-2400) / 4) + 1200, 9000/4), ((((16000-2400) / 4) * 3) + 1200, (9000/4) * 3)),
                      ((3431, 7216), (9420, 7250), (5996, 4229), (14669, 1382)),
                      ((6544,7845), (7489, 1347), (12727, 7116), (4066, 4645), (13010, 1883))]
    
    def __init__(self, targets = 4, targetlist = None, players = 2, save = True):
        self.now = time.time()
        self.save = save
        self.playerCount = players
        self._display_surf = None
        self._running = True
        self.colors = Colors()
        self.caption = "Racer2"
        self.screenId = 0
        self.activeDisplay = 0
        self.window = Window()
        self.field = Field(self, self.window)
        self.audio = Audio()
        self.interface = Interface(self)
        self.totalTargetDist = 0

        ''' Manual Agent'''
        self.game = None

        self.agents = []

        self.inputWidth = 0
        self.drawBrainBig = False
        
        self.play = True
        self.stepMode = False
        self.debug = True
        self.hidden = False
        self.games = 1
        self.tick = 0
        self.laps = 3
        self.windowCopy = None
        self.totalGenerations = 175

        self.tests = 0
        self.maxTests = 20
        self.trainingStage = 0

        self.fps = 30

        self.surfaceOptions = pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.RESIZABLE

        self.loadAttributes()

        self.generateTargets(targets, targetlist)


        self.audio.load()

        self.highScore = 0

    def loadAttributes(self):
        self.scoreboard = [{"id": 0, "highScore": 0}]
        self.trend = []
        self.highscore = 0
        self.hsg = 0
        self.highestCheckpoint = 0
        self.rounds = 1
        self.highestLap = 0
        self.clensed = False
        self._saveFlag = False

    def generatePlayers(self):
        self.players = []
        for p in range(self.playerCount):
            self.players.append(Player(p, self.targets[0].center[0], self.targets[0].center[1], self, self.field))


    def getRandCord(self):
        return (random.randint(1200, self.field.grid.width - 1200), random.randint(1200, self.field.grid.height - 1200))

    def generateTargets(self, targets = 4, targetlist = None):
        self.targets = []
        if targetlist != None:
            targets = len(self.targetProfiles[targetlist])
        for i in range(targets):
            if targetlist == None:
                location = self.getRandCord()
                if len(self.targets) > 1:
                    safe = False
                    while not safe:
                        for target in self.targets:
                            foundIssue = False
                            if distance(location, target.center) < 3000:
                                location = self.getRandCord()
                                foundIssue = True
                                break
                        if foundIssue == False:
                            safe = True
                self.targets.append(Target(self.field, location[0], location[1]))
            else:
                self.targets.append(Target(self.field, self.targetProfiles[targetlist][i][0], self.targetProfiles[targetlist][i][1]))
        
        self.indexTargets()

    def indexTargets(self):
        self.totalTargetDist = 0
        for i, t in enumerate(self.targets):
            n = (i+1) if (i+1) < len(self.targets) else 0
            p = (i-1) if (i-1 ) >= 0 else len(self.targets) -1
            
            t.nIdx = n
            t.pIdx = p
            t.next = self.targets[n]
            t.prev = self.targets[p]
            t.path = angle(t.prev.renderCenter, t.renderCenter)
            t.dist = distance(t.center, t.prev.center)

            self.totalTargetDist += (t.dist - t.size)

    def reinit(self, targetList=None):
        self.loadAttributes()

        if targetList is not None:
            self.generateTargets(None, targetList)
        else:
            self.generateTargets(random.randint(3, 8))

        for p in self.players:
            p.rounds = 1
            p.onTop = 0
            # p.highScore = 0
            p.topScore = 0
            p.scores = {"distance": 0, "accrued": 0}
            p.reinit()

        self.interface.called = False
            
            

    def loadFonts(self):
        self.fonts = pygame.font.get_fonts()
        if "impact" in self.fonts:
            self.font = pygame.font.SysFont("impact", 24, bold=False, italic=False)

    def updateField(self):
        self.field.updateSize()
        for target in self.targets:
            target.fieldTransform()

    def on_init(self):
        pygame.init()

        #pygame.FULLSCREEN
        self._display_surf = pygame.display.set_mode(self.window.size(), self.surfaceOptions, 32, 0, True)
        pygame.display.set_caption(self.caption)
        self.field.loadImages()
        self.updateField()

        self._running = True
        self.audio.play()
        self.loadFonts()

        self.inputs = Inputs(self)
        self.generatePlayers()


    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False
        if event.type == pygame.VIDEORESIZE:
            self._display_surf = pygame.display.set_mode((event.w, event.h), self.surfaceOptions, 32, 0, True)
            self.window.updateSize(event.w, event.h)
            self.updateField()
            self.indexTargets()
            self.inputs.defineInputs()
            pygame.display.set_caption(self.caption)

        if event.type == pygame.WINDOWDISPLAYCHANGED:
            self.screenId = int(event.display_index)

        if self.inputs.active == None:
            if event.type == pygame.KEYUP:
                self.inputs.defineInputs()

                if event.key == pygame.K_ESCAPE:
                    self._running = False
                if event.key == pygame.K_h:
                    self.hidden = not self.hidden
                if event.key == pygame.K_d:
                    self.debug = not self.debug
                    if not self.debug:
                        self.stepMode = False
                if event.key == pygame.K_c:
                    self.stepMode = False
                    self.play = True
                if event.key == pygame.K_m:
                    self.audio.mute()
                if event.key == pygame.K_F5:
                    self.reinit()
                if event.key == pygame.K_b:
                    self._saveFlag = not self._saveFlag
                    print(f"Save Enabled: {self._saveFlag}")
        else:
            if event.type == pygame.KEYDOWN:
                # Check for backspace 
                if event.key == pygame.K_BACKSPACE: 
                    # get text input from 0 to -1 i.e. end. 
                    self.inputs.entry = self.inputs.entry[:-1]
                elif event.key == pygame.K_RETURN or event.key == pygame.K_KP_ENTER:
                    for i in range(len(self.inputs.elements)):
                        if i == self.inputs.active:
                            self.inputs.active = None
                            self.inputs.save(i)
    
                # Unicode standard is used for string 
                # formation 
                else: 
                    self.inputs.entry += event.unicode

        if event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                found = False
                mouse = pygame.mouse.get_pos()

                if self.drawBrainBig or insideRect(mouse, (self.window.width - 200, self.field.height+self.field.border, 200, self.window.height - self.field.height+self.field.border)):
                    self.drawBrainBig = not self.drawBrainBig
                for player in self.players:
                    player.targeted = False
                    if not found:
                        if len(player.frames):
                            center = player.frames[0]["center"]
                        else:
                            center = player.renderCenter
                        if insideCircle(center, mouse, 200 * self.field.gr[1]):
                            player.targeted = True
                            found = True
                            break
                        

                for i, e in enumerate(self.inputs.elements):
                    if insideRect(mouse, e["rect"]):
                        attrib = getattr(self, self.inputs.vaiables[i], None)
                        if callable(attrib):
                            attrib()
                        elif isinstance(attrib, bool):
                            setattr(self, self.inputs.vaiables[i], not attrib)
                        else:             
                            self.inputs.active = i           
                            self.inputs.load(i)
                        break
                    if i == self.inputs.active:
                        self.inputs.active = None
                        self.inputs.save(i)
            if event.button == 2:
                r = Tk()
                r.withdraw()
                r.clipboard_clear()
                o = []
                for t in self.targets:
                    o.append(t.center)
                r.clipboard_append(str("(" + ", ".join(str(v) for v in o) + ")"))
                r.update() # now it stays on the clipboard after the window is closed
                r.destroy()

    def on_input(self):
        key = pygame.key.get_pressed()
      
        if key[pygame.K_s]:
            if self.debug == True and not self.stepMode:
                self.stepMode = True
            self.windowCopy = None
            self.play = True
        

    def drawGame(self):
        self.framesRemaining = 0
        if len(self.scoreboard) >= self.playerCount - 1:
            loop = list(reversed(self.scoreboard))
            t = 0
        else:
            loop = list(reversed(self.players))
            t = 1
        for playerId in range(len(loop)):
            if t:
                player = self.players[loop[playerId].id]
            else:
                player = self.players[loop[playerId]["id"]]

            # print(player.frames)
            if len(player.frames):
                
                color = (int(player.agent.genes[-32:-25], 2), int(player.agent.genes[-40:-33], 2), int(player.agent.genes[-48:-41], 2))
                color2 = (int(player.agent.genes[-56:-49], 2), int(player.agent.genes[-64:-57], 2), int(player.agent.genes[-72:-65], 2))
                color3 = self.colors.targetColors[player.checkpoint]

                targetId = 0 if self.scoreboard[0]["id"] != self.playerCount - 1 else 1
                if player.rank == 0:
                    color = self.colors.white
                if player.id == self.scoreboard[targetId]["id"]:
                    color = self.colors.orange

                frame = player.frames.pop()
                cwidth = 1 
                if player.targeted:
                    cwidth = 0
                if player.rank <= 1:
                    cwidth = 4
                pygame.draw.circle(self._display_surf, color3, frame["center"], 200 * (self.field.gr[1] * 2), width=cwidth)
                
                
                pygame.gfxdraw.filled_polygon(self._display_surf, frame["underPoints"], color2)
                pygame.gfxdraw.aapolygon(self._display_surf, frame["underPointsUnder"], self.colors.black)
                pygame.gfxdraw.filled_polygon(self._display_surf, frame["points"], color)
                pygame.gfxdraw.aapolygon(self._display_surf, frame["pointsUnder"], self.colors.black)
                pygame.gfxdraw.filled_polygon(self._display_surf, frame["points"][1:], color2)
                pygame.gfxdraw.aapolygon(self._display_surf, frame["pointsUnder"][1:], self.colors.black)
                
                pygame.draw.circle(self._display_surf, self.colors.targetColors[player.highestCheckpoint], frame["center"], 40 * (self.field.gr[1] * 2), width=0)

            if len(player.frames):
                self.framesRemaining = len(player.frames)

        if self.framesRemaining:
            return
        self.framesRemaining = 0

    def drawInputs(self):
        self.inputWidth = 0
        for i, s in enumerate(self.inputs.strings):
            t = self.font.render(self.inputs.fstr(s, self.inputs.vaiables[i] if self.inputs.active != i else self.inputs.entry), True, self.colors.white, self.colors.red if self.inputs.active == i else self.colors.dark_gray)
            pygame.draw.rect(self._display_surf, self.colors.red, self.inputs.elements[i]["rect"], width=0)

            inputWidth = self.inputs.elements[i]["rect"][0] + self.inputs.elements[i]["rect"][2]
            if self.inputWidth < inputWidth:
                self.inputWidth = inputWidth
        
            self._display_surf.blit(t, (self.inputs.elements[i]["position"][0], self.inputs.elements[i]["position"][1]))

    def draw_ui(self):
        self._display_surf.fill(self.colors.black)
        pygame.draw.rect(self._display_surf, self.colors.grass, self.field.fieldRect, width=0)
        if self.field.image is not None:
            self._display_surf.blit(self.field.image, (self.field.fieldRect[0], 1 - (self.field.image.get_height() - self.field.fieldRect[3])))
        self.drawTargets()

        self.drawGame()

        # self.field.animate()

        if self.debug:
            self.draw_debug()

        self.drawInfo()        
        self.drawBrain(self.drawBrainBig)
    
    def drawBrain(self, big = False):

        surface = self._display_surf.convert_alpha()

        if big:
            height = self.window.height
            width = self.window.width - 40
            left = 20
            top = 20
            surface.fill([0,0,0,int(255*0.2)])
        else:
            height = self.window.height - self.field.height+self.field.border
            width = 200
            left = self.window.width - 200
            top = self.field.height+self.field.border
            surface.fill([0,0,0,0])
        
        rows = max(len(self.players[0].agent.brain.structure[0]["weights"]), len(self.players[0].agent.brain.structure[1]["weights"]))
        cwidth = width / (self.players[0].agent.genetics.layers + 1)
        coffset = (cwidth / 2) + left
        rheight = height / rows
        hoffset = (rheight / 4) + top

        brainId = 0 if self.scoreboard[0]["id"] != self.playerCount - 1 else 1
        brain = self.players[self.scoreboard[brainId]["id"]].agent.genetics.structure

        rect = (left, top, width, height)
        pygame.draw.rect(surface, ([0,0,0,int(255*0.2)]), rect, width=0)

        if big:
            text = self.font.render(f'Network Agent ID: {self.scoreboard[brainId]["id"]}', True, self.colors.orange)
            textSize = self.font.size(f'Network Agent ID: {self.scoreboard[brainId]["id"]}')
            surface.blit(text, tuple((((self.window.width / 2 ) - (textSize[0] / 2)), textSize[1] * 2)))
        for l in range(len(brain)):

            for wr in range(len(brain[l]["weights"])):
                nodes = len(brain[l]["weights"][0])
                topNode = (rows - nodes) / 2
                nodes = len(brain[l]["weights"])
                topNode2 = (rows - nodes) / 2

                nodePosition2 = (int(coffset + (cwidth * ((l * 2)))), int(hoffset + ((rheight * wr)) + (topNode2 * rheight)))

                for w in range(len(brain[l]["weights"][wr])):
                    nodePosition = (int(coffset + (cwidth + ((l * cwidth)))), int(hoffset + ((rheight * w)) + (topNode * rheight)))
                    nodePosition3 = (int(coffset + (cwidth * l)), int(hoffset + ((rheight * wr)) + (topNode2 * rheight)))

                    lwidth = math.ceil(((brain[l]["weights"][wr][w] + 1) / 2) * 3)
                    if l == 0:
                        pygame.draw.line(surface, self.colors.tgreen if brain[l]["weights"][wr][w] > 0 else self.colors.tred , nodePosition, nodePosition2, lwidth)
                    else:
                        pygame.draw.line(surface, self.colors.tgreen if brain[l]["weights"][wr][w] > 0 else self.colors.tred , nodePosition, nodePosition3, lwidth)
                    


        for l in range(len(brain)):

            for wr in range(len(brain[l]["weights"])):
                nodes = len(brain[l]["weights"][0])
                topNode = (rows - nodes) / 2
                nodes = len(brain[l]["weights"])
                topNode2 = (rows - nodes) / 2

                nodePosition2 = (int(coffset + (cwidth * ((l * 2)))), int(hoffset + ((rheight * wr)) + (topNode2 * rheight)))

                for w in range(len(brain[l]["weights"][wr])):
                    nodePosition = (int(coffset + (cwidth + ((l * cwidth)))), int(hoffset + ((rheight * w)) + (topNode * rheight)))
                    nodePosition3 = (int(coffset + (cwidth * l)), int(hoffset + ((rheight * wr)) + (topNode2 * rheight)))

                    if wr == 0:
                        result = 0
                        # print(f'{self.players[self.scoreboard[brainId]["id"]].agent.brain.results=}')
                        if isinstance(self.players[self.scoreboard[brainId]["id"]].agent.brain.results[l][0], np.float64):
                            result = self.players[self.scoreboard[brainId]["id"]].agent.brain.results[l][w]
                        else:
                            if len(self.players[self.scoreboard[brainId]["id"]].agent.brain.results[l][0]) > w:

                                result = self.players[self.scoreboard[brainId]["id"]].agent.brain.results[l][0][w]
                        
                        pygame.draw.circle(surface, self.colors.tgreen if sigmoid(result) >= 0.5 else self.colors.tred, nodePosition, 36 if big else 6 , width=0)

                        if big:
                            text = self.font.render(f'{sigmoid(result):.3f}', True, self.colors.tblack)
                            textSize = self.font.size(f'{sigmoid(result):.3f}')
                            surface.blit(text, tuple(((nodePosition[0] - (textSize[0]/2)), (nodePosition[1] - (textSize[1]/2)))))

                if l == 0:

                    pygame.draw.circle(surface, self.colors.twhite, nodePosition2, 36 if big else 6, width=0)

                    if big:
                        text = self.font.render(f'{self.players[self.scoreboard[brainId]["id"]].agent.brain.inputs[wr]:.3f}', True, self.colors.tblack)
                        textSize = self.font.size(f'{self.players[self.scoreboard[brainId]["id"]].agent.brain.inputs[wr]:.3f}')
                        surface.blit(text, tuple(((nodePosition2[0] - (textSize[0]/2)), (nodePosition2[1] - (textSize[1]/2)))))
                    
                
        self._display_surf.blit(surface, (0, 0))
    
    def drawTargets(self):
        for i, t in enumerate(self.targets):
            text = self.font.render(str(i), True, self.colors.white, self.colors.black)
            textSize = self.font.size(str(i))
            pygame.draw.circle(self._display_surf, self.colors.targetColors[i], t.renderCenter, t.size * self.field.gr[1], width=2)
            self._display_surf.blit(text, tuple(((t.renderCenter[0] - (textSize[0]/2)), (t.renderCenter[1] - (textSize[1]/2)))))

    def drawProgressChart(self):
        rect = (self.inputWidth + 3, self.field.height + 2, ((self.window.width - 200) - self.inputWidth) - 6, (self.window.height - self.field.height) - 4)
        self.trend.append({"high": self.scoreboard[0 if self.scoreboard[0]["id"] != self.playerCount - 1 else 1], "low": self.scoreboard[len(self.scoreboard)-1]})
        records = len(self.trend)

        if records > rect[2]:
            del self.trend[0]
            records -= 1
 
        pygame.draw.rect(self._display_surf, self.colors.dark_green, rect, width=1)

        if records == 1:
            center = (rect[0] + (rect[2] / 2), rect[1] + (rect[3] / 2))
            pygame.draw.circle(self._display_surf, self.colors.white, center, 2, width=0)
        elif records == 0:
            pass        
        else:
            topTrend = sorted(self.trend, key=lambda r: r["high"]["highScore"], reverse=False).pop()
            lowTrend = sorted(self.trend, key=lambda r: r["low"]["highScore"], reverse=True).pop()
            High = topTrend["high"]["highScore"]
            Low = topTrend["low"]["highScore"]
            if lowTrend["high"]["highScore"] == 0:
                lowTrend["high"]["highScore"] = 1
            if lowTrend["low"]["highScore"] == 0:
                lowTrend["low"]["highScore"] = 1
            for r in range(records - 1):
                r += 1

                row1 = (rect[3] - (rect[3] * (1 * (self.trend[r-1]["high"]["highScore"] / topTrend["high"]["highScore"])))) + rect[1]
                row2 = (rect[3] - (rect[3] * (1 * (self.trend[r]["high"]["highScore"] / topTrend["high"]["highScore"])))) + rect[1]
                row3 = (rect[3] - (rect[3] * (1 * (self.trend[r]["low"]["highScore"] / topTrend["high"]["highScore"])))) + rect[1]
                row4 = (rect[3] - (rect[3] * (1 * (self.trend[r-1]["low"]["highScore"] / topTrend["high"]["highScore"])))) + rect[1]
                left1 = ((rect[2] / records) * r-1) + rect[0]
                left2 = ((rect[2] / records) * r) + rect[0]

                point1 = (left2, row2)
                point2 = (left1 - (rect[2] / records), row1)
                point3 = (left2, row3)
                point4 = (left1 - (rect[2] / records), row4)

                if r > 1:
                    pygame.draw.line(self._display_surf, self.colors.green if self.trend[r-1]["high"]["highScore"] <= self.trend[r]["high"]["highScore"] else self.colors.red , point1, point2, 2)
                    pygame.draw.line(self._display_surf, self.colors.green if self.trend[r-1]["low"]["highScore"] <= self.trend[r]["low"]["highScore"] else self.colors.red , point3, point4, 2)
                pygame.draw.line(self._display_surf, self.colors.green if self.trend[r-1]["low"]["highScore"] <= self.trend[r]["low"]["highScore"] else self.colors.red , point3, point1, 2)
                pygame.draw.circle(self._display_surf, self.colors.green if self.trend[r-1]["high"]["highScore"] <= self.trend[r]["high"]["highScore"] else self.colors.red, point1, 2, width=0)
                pygame.draw.circle(self._display_surf, self.colors.green if self.trend[r-1]["low"]["highScore"] <= self.trend[r]["low"]["highScore"] else self.colors.red, point3, 2, width=0)


            surface = self._display_surf.convert_alpha()
            surface.fill((0,0,0,0))
            text = self.font.render(f'{High=}', True, self.colors.twhite)
            textSize = self.font.size(f'{High=}')
            surface.blit(text, tuple(((((rect[2]/2) + rect[0]) - (textSize[0]/2)), rect[1])))

            text = self.font.render(f'{Low=}', True, self.colors.twhite)
            textSize = self.font.size(f'{Low=}')
            surface.blit(text, tuple(((((rect[2]/2) + rect[0]) - (textSize[0]/2)), (rect[1] + rect[3]) - textSize[1])))

            self._display_surf.blit(surface, (0, 0))



    def drawInfo(self):
        pygame.draw.rect(self._display_surf, self.colors.dark_green, self.field.borderRect, width=self.field.border)
        pygame.draw.rect(self._display_surf, self.colors.dark_gray, (0, 0, self.window.width - self.field.borderRect[2]+self.field.border, self.field.height+self.field.border), width=0)
        pygame.draw.rect(self._display_surf, self.colors.dark_gray, (0, self.field.height+self.field.border, self.window.width, self.window.height-self.field.borderRect[3]+self.field.border), width=0)

        self.drawInputs()

        if isinstance(self.scoreboard, list):
            self.drawScoreboard()
            self.drawProgressChart()
            topId = self.scoreboard[0]["id"]
            text = self.font.render(f'Games: {self.games} ' \
                                    f'Rounds: {self.tests+1} '+ \
                                    f'Lap: {self.players[topId].lap}/3 ' +\
                                    f'Checkpoint: {self.players[topId].checkpoint} ' +\
                                    f'Highscore: {self.highScore} (Gen: {self.hsg}) ' +\
                                    f'Generations: {self.totalGenerations} ' +\
                                    f'geneImpact: {(self.geneImpact * max(0.0001, ((1 / self.totalGenerations) / 4))):0.4f} '+ \
                                    f'Leader: (X: {int(self.players[topId].agent.data["lastMoveReq"][0])} ' +\
                                    f'Y: {int(self.players[topId].agent.data["lastMoveReq"][1])} ' +\
                                    f'Thrust: {self.players[topId].agent.data["lastMoveReq"][2]}) ', +\
                                    True, self.colors.white, self.colors.black)
        else:
            text = self.font.render(f'Lap: {self.players[0].lap}/3      ' +\
                f'Checkpoint: {self.players[0].checkpoint}      ' +\
                f'Generations: {self.players[0].generations}', \
                True, self.colors.white, self.colors.black)

        self._display_surf.blit(text, tuple(((self.field.left), (self.field.top))))
        
    def renderSaving(self, id):
        text = f'Saved {id}'
        t = self.font.render(text, True, self.colors.red, self.colors.dark_gray)
        textSize = self.font.size(text)
        self._display_surf.blit(t, tuple((0, (self.window.width - textSize[1]))))
        
    def drawScoreboard(self):
        lastScore = 0
        printed = 0
        self.updateScoreboard()
        for score, r in enumerate(self.scoreboard):
            if r["highScore"] == lastScore:
                continue

            lastScore = r["highScore"]+r["totalScore"]
            text = f'{str(score)+":":<4} {r["id"]:<3}G{self.players[r["id"]].generations}'
            t = self.font.render(text, True, self.colors.white, self.colors.dark_gray)
            textSize = self.font.size(text)
            self._display_surf.blit(t, tuple((4, (textSize[1]*printed))))

            text = f'{r["highScore"]+r["totalScore"]:>5d}'
            t = self.font.render(text, True, self.colors.white, self.colors.dark_gray)
            textSize = self.font.size(text)
            self._display_surf.blit(t, tuple((self.field.left - (textSize[0] + 8), (textSize[1]*printed))))

            printed += 1

            if textSize[1]*printed > self.window.height:
                break

    def draw_debug(self):
        surface = self._display_surf.convert_alpha()
        surface.fill([0,0,0,0])
        for player in self.players:
            if player.rank < 10:
                pygame.draw.line(surface, self.colors.twhite, player.renderCenter, player.renderTarget, 2)

        for t, target in enumerate(self.targets):
            t = t-1 if t > 0 else len(self.targets)-1
            x = int(int(target.dist * (math.cos(target.path * (math.pi / 180)))) + self.targets[t].renderCenter[0])
            y = int(int(target.dist * (math.sin(target.path * (math.pi / 180)))) + self.targets[t].renderCenter[1])

            pygame.draw.line(surface, self.colors.red, self.targets[t].renderCenter, (x, y), 2)

        self._display_surf.blit(surface, (0, 0))

    def updateScoreboard(self):
        rankedPlayers = [{"id": p.id, "totalScore": p.totalScore, "highScore":  p.highScore, "generation": p.generations, "distance": p.scores["distance"]} for p in self.players]
        rankedPlayers.sort(key=lambda r: r["highScore"] + r["totalScore"], reverse=True)
        topRankedPlayers = list(filter(lambda d: d['highScore'] == rankedPlayers[0]['highScore'], rankedPlayers))
        if len(topRankedPlayers) > 1:
            topRankedPlayers.sort(key=lambda r: r["generation"])
            for i, r in enumerate(topRankedPlayers):
                rankedPlayers[i] = r

        if rankedPlayers[0]["highScore"] > self.highscore:
            self.highscore = rankedPlayers[0]["highScore"]+rankedPlayers[0]["totalScore"]
            self.hsg = self.players[rankedPlayers[0]["id"]].generations

        for i, rating in enumerate(rankedPlayers):
            self.players[rating["id"]].rank = i
            rating["rank"] = i

        self.scoreboard = rankedPlayers

    def on_loop(self):
        if not self.framesRemaining:
            if self.frameDivisor > 1:
                if self.tick % 6 == 0:
                    self.agent_loop()
            else:
                self.agent_loop()

        self.draw_ui()

    # This becomes the main function on codingamge.com mad-pod-racing (Python)
    def agent_loop(self):
        if not self.interface.called:
            del self.game
            self.game = Game()
            laps = (self.interface.input())
            checkpointCount = int(self.interface.input())

            self.game.data['laps'] = laps
            self.game.data['boost'] = True
            self.game.data['tick'] = 0
            self.game.data['checkpointCount'] = checkpointCount
            checkpoints = []
            if 'checkpoint' in self.game.data:
                del self.game.data['checkpoint']
            for i in range(checkpointCount):
                c_x, c_y = [int(j) for j in self.interface.input().split()]
                cp = (c_x, c_y)
                checkpoints.append(cp)

                self.game.createItem('checkpoint', cp)
            loadintUpdate = 0
            for i in range(len(self.players)):
                if len(self.agents) < len(self.players):
                    self.agents.append(Agent(self, laps, checkpointCount, checkpoints))
                    # Remove this on Codingame....
                    loadintUpdate = self.renderLoading(f"Generating Players", i/self.playerCount, loadintUpdate, (63, 122, 43))
                    self.players[i].agent = self.agents[i]
                    self.agents[i].player = self.players[i]
                    self.agents[i].genetics.id = self.players[i].id
                    # ... End Remove

                else:
                    self.agents[i].reinit(laps, checkpointCount, checkpoints)

        # Refacotr in Codingame
        for player in self.players:
            if self.tick >= 1 and player.alive:
                player.housekeeping()
            # print(f'Player ID: ({player.id}) calling Command')
            x, y, vx, vy, a, ncid = [int(j) for j in self.interface.input().split()]
            ncid = ncid % len(self.targets)
            if player.id == self.playerCount - 1:
                self.game.data['tick'] += 1
                if not 'racer' in self.game.data or ('racer' in self.game.data  and 0 > len(self.game.data['racer'])-1):
                    self.game.createEntity('racer', dict(zip("location, direction, angle, ncid, lap, earlyTurn, abortEarlyTurn".replace(" ", "").split(','), [(x, y), (vx, vy), 180-a, ncid, 1, False, False])))
                else:
                    self.game.data['racer'][0].update(dict(zip("location, direction, angle, ncid, lap, earlyTurn, abortEarlyTurn".replace(" ", "").split(','), [(x, y), (vx, vy), 180-a, ncid, self.game.data['racer'][0].data['lap'], self.game.data['racer'][0].data['earlyTurn'], self.game.data['racer'][0].data['abortEarlyTurn']])))
                player.agent.roundData(x, y, vx, vy, a, ncid, self.agents)
            else:
                player.agent.roundData(x, y, vx, vy, a, ncid, self.agents)

        for player in self.players:
            self.interface.command(player.agent.getMove())

    def on_render(self, tick = 30):
        pygame.display.flip()
        pygame.time.Clock().tick(tick)

    def on_cleanup(self):
        self.players[0].genetics.saveStructure("on_exit")
        pygame.quit()
 
    def on_execute(self):
        if self.on_init() == False:
            self._running = False

        lastUpdate = 0
        self.playing = False
        self.training = True
        
        while self._running:
            if self.trainingStage == 2:
                self.maxTests = 10
            elif self.trainingStage == 1:
                self.maxTests = 15
            elif self.trainingStage == 0:
                self.maxTests = 20

            while self._running and (self.training or self.playing):
                for event in pygame.event.get():
                    self.on_event(event)
                self.on_input()
                self.audio.play()
                if self.play == True:
                    self.on_loop()
                    self.on_render()
                    self.tick += 1
                    if self.stepMode == True:
                        self.play = False
                else:
                    if self.windowCopy == None:
                        self.windowCopy = self._display_surf.copy()
                    self.playerDebug()

                if not len([p for p in self.players if p.alive == True]):
                    self.training = False
                    self.rounds += 1
                self.players.sort(key=lambda p: p.highScore, reverse=True)
                for idx, player in enumerate(self.players):
                    player.rank = idx


            # print("Everyone's dead...")
            if self.tests == 0 :
                o = []
                for t in self.targets:
                    o.append(t.center)

                self.targetProfiles[0] = o
            self.tests += 1
            
            
            self.winners = 10

            for idx, player in enumerate(self.players):
                player.highScore += player.totalScore
                if self.tests > 0:
                    player.highScore = int(player.highScore / 2)
            self.players.sort(key=lambda p: p.highScore, reverse=True)

            for idx, player in enumerate(self.players):
                player.rank = idx
                if idx >= self.winners:
                    player.agent.genetics.alter(1, player.agent.genetics.structure)
                    player.agent.brain = player.agent.genetics


            
            counts = {"winners": 0, "direct_children": 0, "mutated_children": 0, "new": 0}

            if self.tests >= self.maxTests:
                # self.players.sort(key=lambda p: p.highScore, reverse=True)
                self.playerSorted = True
                print(f'Generation: {self.totalGenerations} Highscore: {self.players[0].highScore}')
                offspring = []
                for idx, player in enumerate(self.players):
                    if self.tests == self.maxTests and idx == 0:
                        if player.highScore > self.highScore or self._saveFlag:
                            if player.highScore > self.highScore:
                                self.highScore = player.highScore
                                self.hsg = self.totalGenerations
                            loadintUpdate = self.renderLoading(f"Saving new best...", 0, 0, (63, 122, 43))
                            player.agent.genetics.saveStructure(None, player.highScore)
                            loadintUpdate = self.renderLoading(f"Saving new best...", 1, 0, (63, 122, 43))
                    if idx < self.winners:
                        counts["winners"] += 1
                        loadintUpdate = self.renderLoading(f"Spawing Decendents", counts["winners"]/self.winners, loadintUpdate, (63, 122, 43))

                        for pidx in range(idx + 1, self.winners):
                            if idx != pidx:
                                # print(f'Breeding {idx=} with {pidx=}')
                                offspring.append(player.agent.genetics.mate(self.players[pidx]))
                                # print(f"\tResults: {offspring=}")

                    else:  
                        # print(f'{idx - self.winners=}')
                        # print(f'{len(offspring)-1=}')
                        # print((idx - self.winners) % (len(offspring)))
                        loadintUpdate = self.renderLoading(f"Mutating Decendents", (idx-self.winners)/(self.playerCount-self.winners), loadintUpdate, (63, 122, 43))

                        player.agent.genetics.genes = offspring[(idx - self.winners) % (len(offspring ))][(idx % 2) + 1]['genes']
                        if idx >= (len(offspring) * 2) + self.winners:
                            if idx >= (len(offspring) * 4) + self.winners:
                                counts['new'] += 1
                                player.agent.genetics.genGenes()
                                player.agent.genetics.mutate()
                                # player.agent.genetics.genStructure()
                                player.agent.genetics.alter(1, offspring[(idx - self.winners) % (len(offspring ))][(idx % 2) + 1]['brain'])
                            else:
                                counts["mutated_children"] += 1
                                player.agent.genetics.mutate()
                                player.agent.genetics.alter(1, offspring[(idx - self.winners) % (len(offspring ))][(idx % 2) + 1]['brain'])
                            
                        else:
                            counts["direct_children"] += 1
                            player.agent.genetics.structure = copy.deepcopy(offspring[(idx - self.winners) % (len(offspring ))][(idx % 2) + 1]['brain'])

                        
                    player.agent.brain = player.agent.genetics
                    player.highScore = 0
                print(f'{sum(counts.values())} contenders form {counts}')
                self.totalGenerations += 1

            if self.tests == self.maxTests:
                self.tests = 0
                self.reinit()
            else:
                self.reinit(0)
            self.training = True


        self.on_cleanup()

    def renderLoading(self, label, percentage, lastUpdate, color=(128, 172, 245)):
        self.now = time.time()
        if self.now  >= lastUpdate + ((60 / (self.fps) / 60) / 4):
            for event in pygame.event.get():
                self.onLodatingEvent(event)
            font = pygame.font.SysFont("impact", 24, bold=False, italic=False)
            self._display_surf.fill((0, 0, 0, 0))

            text = font.render(f'{label}', True, color)
            textSize = font.size(f'{label}')
            center = (self.window.width / 2, self.window.height / 2)
            self._display_surf.blit(text, (center[0] - (textSize[0] / 2), center[1] - (textSize[1] * 2)))

            if percentage:
                pygame.draw.rect(self._display_surf, color, (int(center[0] * 0.2), center[1] + (textSize[1]), int(center[0] * 1.6), 40), width=1, border_radius=5)
                pygame.draw.rect(self._display_surf, (int(color[0]/2), int(color[1]/2), int(color[2]/2)), (int(center[0] * 0.2) + 1, center[1] + int(textSize[1]) + 1, int((center[0] * 1.6) * percentage) - 2, 38), border_radius=5)

            pygame.display.flip()
            time.sleep(0.0025)

            return self.now
        return lastUpdate
    
    def onLodatingEvent(self, event):
        if event.type == pygame.QUIT:
            self.sigKill()
        if event.type == pygame.WINDOWDISPLAYCHANGED:
            self.screenId = int(event.display_index)
        if event.type == pygame.VIDEORESIZE:
            self._display_surf = pygame.display.set_mode((event.w, event.h), self.surfaceOptions, 32, 0, True)
            self.window.updateSize(event.w, event.h)
            self.updateField()
            self.indexTargets()
            self.inputs.defineInputs()
            pygame.display.set_caption(self.caption)

    def playerDebug(self):
        self._display_surf.fill((0, 0, 0))
        self._display_surf.blit(self.windowCopy, (0, 0))

        surface = self._display_surf.convert_alpha()
        surface.fill([0,0,0,128])
        for player in self.players:
            mouse = pygame.mouse.get_pos()
            if len(player.frames):
                center = player.frames[0]["center"]
            else:
                center = player.renderCenter
            if insideCircle(center, mouse, 200 * self.field.gr[1]):
                s = f'{player.id=} {player.rank=} {sum(player.scores.values())=} {player.highScore=} {player.scores["distance"]=}'
                text = self.font.render(str(s), True, self.colors.tred, self.colors.black)
                textSize = self.font.size(str(s))

                
                x = (((center[0]) - ((textSize[0]/2))) - textSize[1])
                x = 5 if x - (textSize[0]/2) < 0 else x
                x = (self.window.width - textSize[0]) - 5 if x + textSize[0] > self.window.width else x
                y = ((center[1]) - ((textSize[1]/2) - textSize[1]))
                y = 5 if y - ((textSize[1]/2) - textSize[1]) < 0 else y
                y = (self.window.height - textSize[0]) - 5 if y + ((textSize[1]/2) - textSize[1]) > self.window.width else y
                

                surface.blit(text, (x, y))
                self._display_surf.blit(surface, (0, 0))
                break

        
        self.on_render(self.fps)


class Root():
    data = {}
    def __init__(self):
        self.data['previous'] = {}

    def toJSON(self):
        return json.dumps(
        self,
        default=lambda o: {o.data}, 
        sort_keys=True,
        indent=4)

    def add(self, k, v):
        try:
            self.data[k].append(v)
        except:
            self.data[k] = [v]

    def update(self, arg, t = None):
        if isinstance(arg, dict) and t == None:
            for k, v in arg.items():
                if k in self.data:
                    self.data['previous'][k] = self.data[k]
                self.data[k] = v
        else:
            if t in self.data:
                self.data['previous'][t] = self.data[t]
            self.data[t] = arg

    def __str__(self):
        return str(self.toJSON())


class Neural():
    
    def __init__(self, structure):
        self.structure = None
        self.structure = structure
        self.results = []
        self.inputs = []
        

    def forward(self, data):
        self.inputs = data
        output = None
        self.results = []
        
        for i, l in enumerate(self.structure):
            if i == len(self.structure)-1:
                z = self.process(l, output)
                self.results.append(z)
                result = z.tolist().pop()
                output = []
                for r in result:
                    output.append(sigmoid(np.tanh(r)))
                
            else:
                if output is None:
                    z = self.process(l, data)
                    self.results.append(z)
                    output = leakyrelu(0.2, z)
                else:
                    z = self.process(l, output)
                    self.results.append(z)
                    result = z.tolist().pop()
                    output = []
                    for r in result:
                        output.append(leakyrelu(0.2, r))
            
        # print(f'{self.results}')
        return output

    
    def process(self, layer, data):
        # print(f'{data=}')
        # print(f'{self.transpose(layer["weights"].tolist())=}')
        # print(f'{layer["weights"].tolist()=}')
        # print(f'{np.dot(data, layer["weights"])=}')
        # print(f'{math.sumprod(data, self.transpose(layer["weights"].tolist())[0])=}')
        # sumprod is not a direct replacement for np.dot, additional research required.
        
        return np.dot(data, layer["weights"]) + layer["biases"]

    # Must be replaced with custom dot product function -- codingame doesn't support numpy
    def dot(self, weights, inputs):
        if len(weights) != len(inputs):
            raise Exception(f'ValueError: shapes ({len(weights)}, {"" if not isinstance(weights[0], list) else len(weights[0])}) and ({len(inputs)}, {"" if not isinstance(inputs[0], list) else len(inputs[0])}) not aligned') 
        
        if not isinstance(inputs[0], list):
            result = []
            for w in weights:
                r = 0
                for i in inputs:
                    r += w * i
                result.push(r)
        else:
            pass
            
    # Must be replaced with custom transpose function -- codingame doesn't support numpy        
    def transpose(self, array):
        return [[array[j][i] for j in range(len(array))] for i in range(len(array[0]))]


    
class Agent(Root):
    
    def __init__(self, app, laps, checkpointCount, checkpoints):
        super(Agent, self).__init__()
        self.data = {}
        self.app = app
        self.data["laps"] = laps
        self.data["checkpointCount"] = checkpointCount
        self.data["checkpoints"] = checkpoints
        self.data['previous'] = {}
        self.lastThrust = 0
        self.hasBoost = True
        self.mass = 1
        self.usedShield = False
        self.usedShieldTurns = 0

        # Replace this with trained neural structure
        self.genetics = Genetic(self.app, 17, 15, 3, 3, self.app.save, None, copy.deepcopy(self.app.agents[0].genetics.structure) if bool(len(self.app.agents)) else None)
        self.brain = Neural(copy.deepcopy(self.genetics.structure))
        self.genes = str(self.genetics.genes)
        self.output = [0, 0]

    def reinit(self, laps, checkpointCount, checkpoints):
        self.data['previous'] = {}
        self.hasBoost = True
        self.data["checkpointCount"] = checkpointCount
        self.data["checkpoints"] = checkpoints
        self.lastThrust = 0
        self.data["laps"] = laps
        self.output = [0, 0]
        self.mass = 1
        self.usedShield = False
        self.usedShieldTurns = 0

        
    def roundData(self, x, y, vx, vy, angle, ncid, pods):

        self.update(dict(zip(
            "x, y, location, vx, vy, angle, ncid, target, pods".replace(" ", "").split(','),
            [x, y, (x, y), vx, vy, angle, ncid, (self.data["checkpoints"][ncid][0], self.data["checkpoints"][ncid][1]), pods]
        )))
        self.updateVector()
        if "angleOffset" not in self.data:
            self.update(self.data['angle'] + self.data['targetAngle'], "angleOffset")

    def updateVector(self):
        dert = deriv(self.data['location'], self.data['target'])
        dertd = deriv(self.data['target'], self.data['location'])

        if 'location' in self.data['previous']:
            der = deriv(self.data['previous']['location'], self.data['location'])
        else: 
            der = deriv(self.data['location'], self.data['location'])

        speed =  math.sqrt(abs(der['dx'] ** 2) + abs(der['dy'] ** 2))

        self.update(math.atan2(dert['dy'], dert['dx']) * (180 / math.pi), "targetAngle")
        self.update(math.sqrt((dertd['dx'] ** 2) + abs(dertd['dy'] ** 2)), "targetDist")

        if (der['dy'] != 0):
            angle = (math.atan2(der['dy'],  der['dx']) * (180 / math.pi)) 
        else:
            if 'vector' in self.data['previous']:
                angle = self.data['previous']['vector']['d']
            else:
                angle = self.data['angle']

        self.update(vector(angle, speed), 'vector')


    def getFollowingCheckpointAngle(self):
        fcid = self.data["ncid"]+1 if self.data["ncid"]+1 < len(self.data["checkpoints"]) else 0

        der = deriv((self.data["checkpoints"][fcid][0], self.data["checkpoints"][fcid][1]), self.data['target'])
        angle = math.atan2(der['dy'], der['dx']) * (180 / math.pi)
        if angle < 0:
            angle = 360 - angle
        return angle
    
    def direction(self, input):
        direction = 0
        if input > + 0.51:
            direction = 1
        if input < 0.49:
            direction = -1

        return direction

    def getMove(self):
        maxDistance = self.app.targets[self.data["ncid"] % len(self.app.targets)].dist
        fcid = self.data["ncid"]+1 if self.data["ncid"]+1 < len(self.app.targets) else 0
        ffcid = fcid+1 if fcid+1 < len(self.app.targets) else 0

        data = [self.data["angle"] / 360, \
                angleDiffrence(self.data["angle"], self.data["targetAngle"]) / 180, \
                self.data["targetAngle"] / 180, \
                self.data["vector"]["d"] / 360, \
                sigmoid(self.data["vector"]["m"] / 5000), \
                sigmoid(self.data["targetDist"] / 5000), \
                self.app.targets[self.data["ncid"]].path / 360, \
                sigmoid(self.app.targets[self.data["ncid"]].dist / 5000), \
                self.app.targets[fcid].path / 360, \
                sigmoid(self.app.targets[fcid].dist / 5000), \
                angleDiffrence(self.app.targets[self.data["ncid"]].path, self.app.targets[fcid].path) / 180, \
                self.app.targets[ffcid].path / 360, \
                sigmoid(self.app.targets[ffcid].dist / 5000), \
                angleDiffrence(self.app.targets[fcid].path, self.app.targets[ffcid].path) / 180, \
                int(self.hasBoost),
                self.output[0],
                self.output[1]]
        output = self.brain.forward(data)
        self.output = output

        thrust = abs(math.ceil(output[1] * 100)) if not output[2] > 0.9 else "BOOST" if output[2] > 0.1 else "SHIELD"
        if thrust == "BOOST":
            if self.hasBoost == True:
                self.hasBoost = False
                thrust = 200
            else:
                thrust = 100
        if thrust == "SHIELD" or self.usedShield:
            thrust = 0
            self.usedShield = True
            self.usedShieldTurns += 1
            self.mass = 10
            if self.usedShieldTurns > 3:
                self.usedShield = False
                self.mass = 1
        direction = self.direction(output[0])

        turnAmount = 0
        if direction > 0:
            turnAmount = max(1, int(direction * (18 * (((output[0] - 0.01) - 0.5) / 0.5))))
        elif direction < 0:
            turnAmount = min(-1, int(direction * (18 * ((0.5 - (output[0] + 0.01)) / 0.5))))
        self.lastThrust = thrust

        # if len(self.app.scoreboard) >= self.app.playerCount -1 and self.player.id == self.app.scoreboard[1]["id"]:
        # print(f'{output[0]=} {direction=} {turnAmount=} {self.data["ncid"]=} {fcid=}')

        x = int(int(thrust * 100) * (math.cos((self.data["angle"] + turnAmount) * (math.pi / 180)))) + self.data["location"][0]
        y = int(int(thrust * 100) * (math.sin((self.data["angle"] + turnAmount) * (math.pi / 180)))) + self.data["location"][1]

        self.update((x, y), "lastMove")
        self.update((x, y, thrust), "lastMoveReq")

        return f'{x} {y} {thrust}'
if __name__ == "__main__" :
    # print(sys.argv[1])
    theApp = App(random.randint(3, 8), int(sys.argv[1]) if len(sys.argv) > 1 else None, 500, save = True)
    theApp.on_execute()