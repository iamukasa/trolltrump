import numpy as np




class RecurrentNeuralNetwork:
    def __init__(self, xs, ys, rl, eo, lr):
        self.x = np.zeros(xs)
        self.xs = xs
        self.y = np.zeros(ys)
        self.ys = ys
        self.w = np.random.random((ys, ys))
        self.G = np.zeros_like(self.w)
        self.rl = rl
        self.lr = lr
        self.ia = np.zeros((rl + 1, xs))
        self.ca = np.zeros((rl + 1, xs))
        self.oa = np.zeros((rl + 1, xs))
        self.ha = np.zeros((rl + 1, xs))
        self.af = np.zeros((rl + 1, xs))
        self.ai = np.zeros((rl + 1, xs))
        self.ac = np.zeros((rl + 1, xs))
        self.ao = np.zeros((rl + 1, xs))
        self.eo = np.vstack((np.zeros(eo.shape[0]), eo.T))
        self.LSTM = LSTM(xs, ys, rl, lr)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def dsigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def forwardProp(self):
        for i in range(1, self.rl + 1):
            self.LSTM.x = np.hstack((self.ha[-1], self.x))
            cs, hs, f, inp, c, o = self.LSTM.forwardProp()
            self.ca[i] = cs
            self.ha[i] = hs
            self.af[i] = f
            self.ai[i] = inp
            self.ac[i] = c
            self.ao[i] = o
            self.oa[i] = self.sigmoid(np.dot(self.w, hs))
            self.x = self.eo[i - 1]
        return self.oa

    def backProp(self):
        totalError = 0

        dfcs = np.zeros(self.ys)
        dfhs = np.zeros(self.ys)
        tu = np.zeros((self.ys, self.ys))
        tfu = np.zeros((self.ys, self.xs + self.ys))
        tiu = np.zeros((self.ys, self.xs + self.ys))
        tcu = np.zeros((self.ys, self.xs + self.ys))
        tou = np.zeros((self.ys, self.xs + self.ys))

        for i in range(self.rl, -1, -1):
            error = self.oa[1] - self.eo[i]
            tu += np.dot(np.atleast_2d(error * self.dsigmoid(self.oa[i])), np.atleast_2d(self.ha[i]).T)

            error = np.dot(error, self.w)

            self.LSTM.x = np.hstack((self.ha[i - 1], self.ia[i]))
            self.LSTM.cs = self.ca[i]
            fu, iu, cu, ou, dfcs, dfhs = self.LSTM.backProp(error, self.ca[i - 1], self.af[i], self.ai[i], self.ac[i],
                                                            self.ao[i], dfcs, dfhs)
            totalError += np.sum(error)
            tfu += fu
            tiu += iu
            tcu += cu
            tou += ou
        self.LSTM.update(tfu / self.rl, tiu / self.rl, tcu / self.rl, tou / self.rl)
        self.update(tu / self.rl)
        return totalError

    def update(self, u):
        self.G = 0.9 * self.G + 0.1 * u ** 2
        self.w -= self.rl / np.sqrt(self.G + 1e-8) * u
        return

    def sample(self):
        for i in range(1, self.rl + 1):
            self.LSTM.x = np.hstack((self.ha[-1], self.x))
            cs, hs, f, inp, c, o = self.LSTM.forwardProp()
            maxI = np.argmax(self.x)
            self.x = np.zeros_like(self.x)
            self.x[maxI] = 1
            self.ia[i] = self.x

            self.ca[i] = cs
            self.ha[i] = hs
            self.af[i] = f
            self.ai[i] = inp
            self.ac[i] = c
            self.ao[i] = o
            self.oa[i] = self.sigmoid(np.dot(self.w, hs))
            maxI = np.argmax(self.oa[i])
            newX = np.zeros_like(self.x)
            newX[maxI] = 1
            self.x = newX
        return self.oa


class LSTM:
    def __init__(self, xs, ys, rl, lr):
        self.x = np.zeros(xs + ys)
        self.xs = xs + ys
        self.y = np.zeros(ys)
        self.ys = ys
        self.cs = np.zeros(ys)
        self.rl = rl
        self.lr = lr
        self.f = np.random.random((ys, xs + ys))
        self.i = np.random.random((ys, xs + ys))
        self.c = np.random.random((ys, xs + ys))
        self.o = np.random.random((ys, xs + ys))
        self.Gf = np.zeros_like(self.f)
        self.Gi = np.zeros_like(self.i)
        self.Gc = np.zeros_like(self.c)
        self.Go = np.zeros_like(self.o)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def dsigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def tangent(self, x):
        return np.tanh(x)

    def dtangent(self, x):
        return 1 - np.tanh(x) ** 2

    def forwardProp(self):
        f = self.sigmoid(np.dot(self.f, self.x))
        self.cs *= f
        i = self.sigmoid(np.dot(self.i, self.x))
        c = self.tangent(np.dot(self.c, self.x))
        self.cs += i * c
        o = self.sigmoid(np.dot(self.o, self.x))
        self.y = o * self.tangent(self.cs)
        return self.cs, self.y, f, i, c, o

    def backProp(self, e, pcs, f, i, c, o, dfcs, dfhs):
        e = np.clip(e + dfhs, -6, 6)
        do = self.tangent(self.cs) * e

        ou = np.dot(np.atleast_2d(do * self.dtangent(o)).T, np.atleast_2d(self.x))
        dcs = np.clip(e * o * self.dtangent(self.cs) + dfcs, -6, 6)
        dc = dcs * i

        cu = np.dot(np.atleast_2d(dc * self.dtangent(c)).T, np.atleast_2d(self.x))
        di = dcs * c

        iu = np.dot(np.atleast_2d(di * self.dtangent(i)).T, np.atleast_2d(self.x))
        df = dcs * pcs

        fu = np.dot(np.atleast_2d(df * self.dtangent(f)).T, np.atleast_2d(self.x))
        dpcs = dcs * f

        dphs = np.dot(dc, self.c)[:self.ys] + np.dot(do, self.o)[:self.ys] + np.dot(di, self.i)[:self.ys] + np.dot(df,
                                                                                                                   self.f)[
                                                                                                            :self.ys]

        return fu, iu, cu, ou, dpcs, dphs

    def update(self, fu, iu, cu, ou):
        self.Gf = 0.9 + self.Gf + 0.1 * fu ** 2
        self.Gi = 0.9 + self.Gi + 0.1 * iu ** 2
        self.Gc = 0.9 + self.Gc + 0.1 * cu ** 2
        self.Go = 0.9 + self.Go + 0.1 * ou ** 2

        self.f -= self.lr / np.sqrt(self.Gf + 1e-8) * fu
        self.i -= self.lr / np.sqrt(self.Gi + 1e-8) * iu
        self.c -= self.lr / np.sqrt(self.Gc + 1e-8) * cu
        self.o -= self.lr / np.sqrt(self.Go + 1e-8) * ou

        return


def LoadText(data):
    # text =data.
    text=data.split()
    outputSize = len(text)
    data = list(set(text))
    uniqueWords, dataSize = len(data), len(data)
    returnData = np.zeros((uniqueWords, dataSize))
    for i in range(0, dataSize):
        returnData[i][i] = 1
    returnData = np.append(returnData, np.atleast_2d(data), axis=0)
    output = np.zeros((uniqueWords, outputSize))
    for i in range(0, outputSize):
        index = np.where(np.asarray(data) == text[i])
        output[:, i] = returnData[0:-1, index[0]].astype(float).ravel()
        return returnData, uniqueWords, output, outputSize, data


def ExportText(output, data):
    finalOutput = np.zeros_like(output)
    prob = np.zeros_like(output[0])
    outputText = ""
    print(len(data))
    print(output.shape[0])
    for i in range(0, output.shape[0]):
        for j in range(0, output.shape[1]):
            prob[j] = output[i][j] / np.sum(output[i])
        outputText += " "+np.random.choice(data, p=prob)
    with open("outputweb.txt", "w") as text_file:
        text_file.write(outputText)
    return outputText


def rundis(userinput):
    generatedresult=""
    # Begin program\n",
    print("Beginning")
    iterations = 5000
    learningRate = 0.001


    # load input output data (words)
    returnData, numCategories, expectedOutput, outputSize, data = LoadText(userinput)
    print("Done Reading")
    # init our RNN using our hyperparams and dataset
    RNN = RecurrentNeuralNetwork(numCategories, numCategories, outputSize, expectedOutput, learningRate)

    b=[]

    # training time!\n",
    for i in range(1, iterations):
        # compute predicted next word
        RNN.forwardProp()
        # update all our weights using our error
        error = RNN.backProp()

        # once our error/loss is small enough\n",

        sophie_brain = open("slog.txt", "w")
        b.append(error)
        sophie_brain.writelines(str(b))
        sophie_brain.close()



        print("Error on iteration ", i, ": ", error),
        if error > -100 and error < 100 or i % 100 == 0:
            # we can finally define a seed word\n",
            seed = np.zeros_like(RNN.x)
            maxI = np.argmax(np.random.random(RNN.x.shape))
            seed[maxI] = 1
            RNN.x = seed
            # and predict some new text!\n",
            output = RNN.sample()
            print(output)
            # write it all to disk\n",
            generatedresult=ExportText(output, data)

            print("Done Writing")


    print("Complete")
    return generatedresult



