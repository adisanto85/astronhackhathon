'''
Created on Apr 23, 2018

@author: ldisantao
'''
import numpy
from matplotlib import pyplot
from sklearn.cluster.k_means_ import KMeans
from mpl_toolkits.mplot3d import Axes3D
import FATS


def calc_hour( str ):
    hour, min, sec = [float(i) for i in str.split(':')]
    min += sec/60.
    hour += min/60.
    return hour

def detectionCounter(runcatid):
    i=0
    counter=1
    counter_list = []
    while i < len(runcatid)-1:
        i = i + 1
        if runcatid[i] == runcatid[i-1]:
            counter = counter + 1
        elif runcatid[i] != runcatid[i-1]:
             counter_list.append(counter)
             counter = 1
    print counter_list
    print numpy.asarray(counter_list).shape
    return counter_list

def plotTimeSeries(data, t, runcatid, start=0):
    i=start
    flux = []
    time = []
    flux.append(data[i])
    time.append(t[i])
    pyplot.figure()
    while i < len(runcatid)-1:
        i = i + 1
        if runcatid[i] == runcatid[i-1]:
            flux.append(data[i])
            time.append(t[i])
        elif runcatid[i] != runcatid[i-1]:
            flux = -2.5*numpy.log10(flux)
            pyplot.plot(time, flux, 'o')
            pyplot.show()
            flux = []
            time = []
            flux.append(data[i])
            time.append(t[i])
            continue

def extractFeatures(coord, data, t, runcatid):
    x, y = coord
    i=0
    featuresTable = []
    flux = []
    time = []
    ra = []
    dec = []
    if numpy.isnan(data[i]):
        flux.append(0.0)
    else:
        flux.append(data[i])
        time.append(t[i])
    while i < len(runcatid)-1:
        i = i + 1
        if numpy.isnan(data[i]):
            flux.append(0.0)
        elif data[i] != numpy.NaN and runcatid[i] == runcatid[i-1]:
            flux.append(data[i])
            time.append(t[i])
        elif data[i] != numpy.NaN and runcatid[i] != runcatid[i-1]:
            try:
                
                mag = -2.5*numpy.log10(flux)
    #             time = numpy.sort(time)
                lc = numpy.array([mag, time])
                a = FATS.FeatureSpace(Data=['magnitude','time'],featureList=['Mean','Std',
                                                        'Eta_e','FluxPercentileRatioMid20', 'FluxPercentileRatioMid35',
                                                        'FluxPercentileRatioMid50','FluxPercentileRatioMid65','FluxPercentileRatioMid80',
                                                        'LinearTrend','MaxSlope','Meanvariance','MedianAbsDev',
                                                        'MedianBRP','PairSlopeTrend','PercentAmplitude','PercentDifferenceFluxPercentile'
                                                        'Skew','VariablityIndex', 'Anderson-Darling test', 'Q31', 'Rcs'])
                a=a.calculateFeature(lc)
                output = a.result(method='array')
#                 print output
                featuresTable.append(output)
                flux = []
                time = []
                flux.append(data[i])
                time.append(t[i])
                ra.append(x[i-1])
                dec.append(y[i-1])
                continue
            except ValueError:
                pass
    
    featuresTable = numpy.asarray(featuresTable)
    ra = numpy.asarray(ra)
    dec = numpy.asarray(dec)

    return featuresTable, ra, dec


calc_hour = numpy.vectorize( calc_hour )



data = numpy.loadtxt('/lhome/ldisantao/workspace/AstronHackathon/data/AARTFAAC-candidates-lc_matching.csv', 
                     delimiter=',', skiprows=1, usecols=[0,1,3,4,5,6])

time = numpy.loadtxt('/lhome/ldisantao/workspace/AstronHackathon/data/AARTFAAC-candidates-lc_matching.csv', 
                     dtype='str', delimiter=',', skiprows=1, usecols=[2])

hours= calc_hour( time )

print data.shape
print time.shape

coord = data[:,0], data[:,1]
flux = data[:,2]
ids = data[:,5]

id_index = numpy.ndarray.tolist(ids).index(9432)
print id_index

# plotTimeSeries(flux, hours, ids, start=id_index)
featuresTable, ra, dec = extractFeatures(coord, flux, hours, ids)
# exit()

where_are_NaNs = numpy.isnan(featuresTable)
featuresTable[where_are_NaNs] = 0

where_are_inf = numpy.isinf(featuresTable)
featuresTable[where_are_inf] = 0

print featuresTable.shape
print ra.shape
print dec.shape

print featuresTable
numpy.savetxt('featuresTable.csv', featuresTable, delimiter=',')


# counter_list = numpy.asarray(detectionCounter(data[:,5]))
# 
# detections = [x for x in counter_list for _ in range(x)]
# detections = numpy.asarray(detections).reshape((-1,1))
# features = data[:,0:3]
# print features.shape
# print detections.shape

# X = numpy.concatenate((features,detections)).reshape((len(detections),4))
# print X.shape

numpy.random.seed(12345)
numpy.random.shuffle(featuresTable)
 
num = len(featuresTable)
 
 
kmeans = KMeans(init='k-means++', n_clusters=5, n_init=10, n_jobs=-1).fit(featuresTable)
 
labels = kmeans.labels_.reshape((-1,1))
fig = pyplot.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(data[:num,0], data[:num,1], zs=hours[:num], zdir='z',c=labels[:num])
pyplot.scatter(ra[:num],dec[:num],c=labels[:num])
# pyplot.scatter(data[:100000,0], data[:100000,1], c=kmeans.labels_,lw=0, s=120)
# ax.view_init(elev=90, azim=40)
# pyplot.savefig("/lhome/ldisantao/workspace/AstronHackathon/plots/features_clustering.png", dpi=150, format='png', bbox_inches='tight')
# for ii in xrange(0,360,20):
#         ax.view_init(elev=ii, azim=40)
#         pyplot.savefig("/lhome/ldisantao/workspace/AstronHackathon/plots/clustering%d.png" % ii, dpi=50, format='png', bbox_inches='tight')
#         print 'new save done'
print 'save finished'
pyplot.show()
# astronhackhathon
