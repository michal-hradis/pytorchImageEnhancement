class PsfNet(nn.Module):

    def __init__(self, input_img, embeddingDim):
        super(PsfNet, self).__init__()

        self.filterSizes = [5, 3, 3]
        self.filterCounts = [24, 48, 64]
        self.strides = [1, 1]
        self.fcSizes = [128, embeddingDim]
        
        lastChannels = input_img.data.shape[1]
        self.convLayers = []
        for fSize, fCount, stride in zip(self.filterSizes, self.filterCounts, self.strides):
            self.convLayers.append(nn.Conv2d(in_channels=lastChannels, 
                                             out_channels=fCount, 
                                             kernel_size=fSize, 
                                             stride=stride))
            lastChannels = fCount
            self.convLayers.append(nn.BatchNorm2d(num_features=lastChannels, momentum=0.5))
            self.convLayers.append(nn.ReLU())
        
        self.convLayers = nn.Sequential(*self.convLayers).cuda()
        out = self.convLayers(input_img)
        print('Conv output shape:', out.data.shape)
        
        lastChannels = out.data.shape[1]*out.data.shape[2]*out.data.shape[3]
        self.fcLayers = []
        for size in self.fcSizes[:-1]:
            self.fcLayers.append(nn.Linear(lastChannels, size))
            lastChannels = size
            #self.fcLayers.append(nn.BatchNorm2d(num_features=lastChannels, momentum=0.3))
            self.fcLayers.append(nn.ReLU())
            
        self.fcLayers.append(nn.Linear(lastChannels, self.fcSizes[-1]))
        lastChannels = self.fcSizes[-1]
           
        self.fcLayers = nn.Sequential(*self.fcLayers).cuda()
        
        self.outBNorm = nn.BatchNorm1d(lastChannels, momentum=0.5)
        
    def forward(self, input_x):
        x = input_x
        x = self.convLayers(x)
        x = x.view(x.data.shape[0], -1)
        x = self.fcLayers(x)
        x = F.normalize(x, p=2, dim=1, eps=1e-12)
        x = self.outBNorm(x)
        return x

class LinearFilterNet(nn.Module):

    def __init__(self, filterSizes, filterCounts):
        super(LinearNet, self).__init__()

        lastChannels = 3
        self.layers = []
        for fSize, fCount in zip(filterSizes, filterCounts):
            self.layers.append(nn.Conv2d(lastChannels, fCount, fSize))
            lastChannels = fCount

        self.layers = nn.ModuleList(self.layers)
                              
    def forward(self, input_x, psf):
        x = input_x
        for l in list(self.layers)[:-1]:
            x = F.tanh(l(x))
        x = self.layers[-1](x)
        
        b = (input_x.data.shape[2] - x.data.shape[2]) / 2
        input_x = input_x[:, :, b:-b, b:-b]
        x = x + input_x 
        
        return x
    
    
class WeightNet(nn.Module):
    def __init__(self, inDim, outDims):
        super(WeightNet, self).__init__()
 
        self.expandModule = []
        self.expandModule.append(nn.Linear(inDim, 128))
        self.expandModule.append(nn.BatchNorm2d(num_features=128, momentum=0.5))
        self.expandModule.append(nn.ReLU())
        self.expandModule = nn.Sequential(*self.expandModule)
        
        self.weightModules = nn.ModuleList()
        for outDim in outDims:
            modul = []
            modul.append(nn.Linear(128, outDim))
            modul.append(nn.Softmax())
            self.weightModules.append(nn.Sequential(*modul))
            
    def forward(self, input_x):
        x = input_x 
        x = self.expandModule(x)
        
        outputs = []
        for modul in self.weightModules:
            outputs.append(modul(x))
        
        return outputs
        
        

class DeconvNet(nn.Module):
    def __init__(self, psfNet, psfDim, filterSizes, filterCounts):
        super(DeconvNet, self).__init__()

        self.psfNet = psfNet
        
        self.weightNet = WeightNet(psfDim, filterCounts)
        
        lastChannels = 3
        self.layers = nn.ModuleList()
        self.BNlayers = nn.ModuleList()
        for fSize, fCount in zip(filterSizes, filterCounts):
            self.layers.append(nn.Conv2d(lastChannels, fCount, fSize))
            lastChannels = fCount
            self.BNlayers.append(nn.BatchNorm2d(num_features=lastChannels, momentum=0.5))

        self.lastLayer = nn.Conv2d(lastChannels, 3, 3)
            
    def forward(self, input_x, psf):
        x = input_x
        emb = self.psfNet(psf)
        weights = self.weightNet(emb)

        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = x * weights[i].view(x.data.shape[0], x.data.shape[1], 1, 1)
            x = self.BNlayers[i](x) 
            x = F.tanh(x)

        x = self.lastLayer(x)
        b = (input_x.data.shape[2] - x.data.shape[2]) / 2
        input_x = input_x[:, :, b:-b, b:-b]
        x = x + input_x 
        
        return x
    
    

class LinearNet(nn.Module):

    def __init__(self, filterSizes, filterCounts):
        super(LinearNet, self).__init__()

        lastChannels = 3
        self.layers = []
        for fSize, fCount in zip(filterSizes, filterCounts):
            self.layers.append(nn.Conv2d(lastChannels, fCount, fSize))
            lastChannels = fCount

        self.layers = nn.ModuleList(self.layers)
                              
    def forward(self, input_x):
        x = input_x
        for l in list(self.layers)[:-1]:
            x = F.tanh(l(x))
        x = self.layers[-1](x)
        
        b = (input_x.data.shape[2] - x.data.shape[2]) / 2
        input_x = input_x[:, :, b:-b, b:-b]
        x = x + input_x 
        
        return x
    
    class Net(nn.Module):

    def __init__(self, ):
        super(Net, self).__init__()
        
        self.embDim = 32
        self.filterSize = 25
        self.filterCount = 3
        
        self.psfEmbed = nn.Embedding(
            maxFilterID, self.filterCount*self.filterSize*self.filterSize*3)#self.embDim)
        #self.psfFC1 = torch.nn.Linear(self.embDim, 64, bias=False)
        #self.psfFC2 = torch.nn.Linear(
        #    64, 
        #    self.filterCount*self.filterSize*self.filterSize*3, 
        #                              bias=False)
        
        
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        #self.conv1 = nn.Conv2d(self.filterCount, 64, 1)
        #self.conv2 = nn.Conv2d(64, 3, 7)
        
    def computeEmb(self, psfIdx):
        batchSize = psfIdx.data.shape[0]
        f = self.psfEmbed(psfIdx)
        #f = f.view(
        #    [batchSize, self.filterCount, self.filterSize*self.filterSize*3]).transpose(0,1).clone()
        
        #f = f.view(f.data.shape[0], f.data.shape[2])
        #f = F.relu(self.psfFC1(f))
        #f = self.psfFC2(f)
        #f = F.softmax(f)
        #f = f.view(batchSize, self.filterCount*3, self.filterSize, self.filterSize)
        
        #f = f.view(batchSize, self.filterCount, 3 , self.filterSize, self.filterSize)
        
        return f
    
    def forward(self, input_x, psf):
        batchSize = psfIdx.data.shape[0]
        
        f = self.computeEmb(psfIdx)
        x = input_x
        res = []
        for i in range(batchSize):
            img = x[i:i+1]
            w = f[i].view(self.filterCount, 3, self.filterSize, self.filterSize)
            res.append(F.conv2d(img, w))
        res = torch.cat(res, dim=0)

        #x = x.view(1, x.data.shape[0]*x.data.shape[1],
        #          x.data.shape[2], x.data.shape[3])
        #x = F.tanh(res)
        #x = x.view(batchSize, self.filterCount, x.data.shape[2], x.data.shape[3])
        
        #x = F.relu(self.conv1(x))
        #x = self.conv2(x)
        
        #b = (input_x.data.shape[2] - x.data.shape[2]) / 2
        #input_x = input_x[:, :, b:-b, b:-b]
        #x = x + input_x 
        
        return res