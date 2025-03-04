import glob
import numpy as np

__macr_names__ = ['phi']
__info__ = dict()


def getFilenamesMacr(macrName,path):
    listFiles = sorted(glob.glob(path + "*" + macrName + "*.bin"))
    return listFiles


def getMacrSteps(path):
    listFiles = getFilenamesMacr(__macr_names__[0],path)
    setMacrSteps = set()
    for i in listFiles:
        macrStep = i.split(__macr_names__[0])[-1]
        macrStep = macrStep[:-4]  
        macrStep = int(macrStep)  
        setMacrSteps.add(macrStep)  
    listMacrSteps = sorted(setMacrSteps)
    return listMacrSteps


def getSimInfo(path):
    if len(__info__) == 0:
        filename = glob.glob(path + "*info*.txt")[0]
        with open(filename, "r") as f:
            lines = f.readlines()
            linesTrim = [l.strip() for l in lines]

            try:
                __info__['ID'] = [str(txt.split(" ")[-1]) for txt in linesTrim
                                  if 'Simulation ID' in txt][0]
            except BaseException:
                print("Not able to get ID from info file")

            try:
                __info__['Prc'] = [
                    txt.split(" ")[-1] for txt in linesTrim if 'Precision' in txt][0]
            except BaseException:
                print("Not able to get Precision from info file")

            try:
                __info__['NX'] = [int(txt.split(" ")[-1])
                                  for txt in linesTrim if 'NX' in txt][0]
            except BaseException:
                print("Not able to get NX from info file")

            try:
                __info__['NY'] = [int(txt.split(" ")[-1])
                                  for txt in linesTrim if 'NY' in txt][0]
            except BaseException:
                print("Not able to get NY from info file")

            try:
                __info__['NZ'] = [int(txt.split(" ")[-1])
                                  for txt in linesTrim if 'NZ:' in txt][0]
            except BaseException:
                print("Not able to get NZ from info file")

            try:
                __info__['NZ_TOTAL'] = [int(txt.split(" ")[-1])
                                  for txt in linesTrim if 'NZ_TOTAL' in txt][0]
            except BaseException:
                print("Not able to get TOTAL_NZ from info file")

            try:
                __info__['Tau'] = [float(txt.split(" ")[-1])
                                   for txt in linesTrim if 'Tau' in txt][0]
            except BaseException:
                print("Not able to get Tau from info file")

            try:
                __info__['Umax'] = [float(txt.split(" ")[-1])
                                    for txt in linesTrim if 'Umax' in txt][0]
            except BaseException:
                print("Not able to get Umax from info file")

            try:
                __info__['Nsteps'] = [int(txt.split(" ")[-1])
                                      for txt in linesTrim if 'Nsteps' in txt][0]
            except BaseException:
                print("Not able to get Nsteps from info file")

            try:
                __info__['FX'] = [float(txt.split(" ")[-1])
                                  for txt in linesTrim if 'FX' in txt][0]
            except BaseException:
                print("Not able to get FX from info file")

            try:
                __info__['FY'] = [float(txt.split(" ")[-1])
                                  for txt in linesTrim if 'FY' in txt][0]
            except BaseException:
                print("Not able to get FY from info file")

            try:
                __info__['FZ'] = [float(txt.split(" ")[-1])
                                  for txt in linesTrim if 'FZ' in txt][0]
            except BaseException:
                print("Not able to get FZ from info file")

    return __info__


def readFileMacr3D(macrFilename,path):
    info = getSimInfo(path)
    if info['Prc'] == 'double':
        prc = 'd'
    elif info['Prc'] == 'float':
        prc = 'f'
    with open(macrFilename, "r") as f:
        vec = np.fromfile(f, prc)
        vec3D = np.reshape(vec, (info['NZ_TOTAL'], info['NY'], info['NX']), 'C')
        return np.swapaxes(vec3D, 0, 2)


def getMacrsFromStep(step,path):
    macr = dict()
    listFilename = list()

    for macrName in __macr_names__:
        listFilename.append(getFilenamesMacr(macrName,path))

    listFilenameFlat = [j for i in range(0, len(listFilename))
                        for j in listFilename[i]]

    listNames = ["%s%06d.bin" % (macr, step) 
                for macr in __macr_names__]

    listFilenameStep = [
        i for i in listFilenameFlat if
            any([True for j in listNames if j in i])]
    if len(listFilenameStep) == 0:
        return None

    for filename in listFilenameStep:
        for macrName in __macr_names__:
            if macrName in filename:
                macr[macrName] = readFileMacr3D(filename,path)

    return macr


def getAllMacrs(path):
    macr = dict()

    filenames = dict()
    for macrName in __macr_names__:
        filenames[macrName] = getFilenamesMacr(macrName,path)

    for i in range(0, min(len(filenames[i]) for i in filenames)):
        macrStep = filenames[__macr_names__[0]][i].split(__macr_names__[0])[-1]
        macrStep = macrStep[:-4]  # take off ".bin"
        macrStep = int(macrStep)

        macr[macrStep] = dict()
        for macrName in filenames:
            macr[macrStep][macrName] = readFileMacr3D(filenames[macrName][i],path)

    return macr
