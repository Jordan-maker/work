@userCreds
def get(args=None):
    manager = Manager()
    validatingType = 'c'

    args.dataset = args.dataset.rstrip('/')
    #if not args.dataset.count('/'):
    #    args.dataset = os.path.join(args.dataset, '*')

    lfns = manager.getLfns(args.subcate, args.dataset, args.user, False, noDirs=True, resolveDatablocks=(not args.noSubDir))

    if args.input_dslist != None:
        lfnsList = open(args.input_dslist, 'r')
        lines = lfnsList.readlines()
        lfns_new = lfns.copy()
        lfns_new['Value'] = []
        for line in lines:
            [lfns_new['Value'].append(line.strip()) for lfn in lfns['Value'] if line.strip() == lfn]
        lfns = lfns_new

    if not lfns['OK']:
        print(lfns['Message'])
        sys.exit(1)
    if len(lfns['Value']) == 0:
        print('No file found')
        sys.exit(0)

    localDir = args.output_dir
    if localDir is None:
        localDir = os.getcwd()

    lfnDict={}
    for lfn in lfns['Value']:
        dname = os.path.basename(os.path.dirname(lfn))

        if '/belle/user' in lfn:
            _lfn = lfn.split('/')[4:]
            dname = '/'.join(_lfn[:-1])        

        dirFile = lfnDict.get(dname, [])
        dirFile.append(lfn)
        lfnDict[dname] = dirFile
    
    validatingType = ''
    if args.force:
        validatingType = 'c'

    if args.failed_lfns and os.path.isfile(args.failed_lfns):
        UI().askRepeated('A file with name "{}" already exists. '.format(args.failed_lfns) \
                        +'Do you want to overwrite it?')
                
    for subDir in sorted(lfnDict):
        _localDir = os.path.join(localDir, subDir)
        try:
            os.makedirs(_localDir)
        except OSError:
            print('%s already exists' % _localDir)
            if not args.force and validatingType == '':
                msg = 'How would you like to verify the files? ' \
                    + 'By file size(s) or by file checksum(c)?: '

                while validatingType == '':
                    ret = UI().twoOptionDialog('s', 'c', msg)
                    if ret['OK']:
                        validatingType = ret['Value']

    if not args.force:
        for subDir in sorted(lfnDict):
            _localDir = os.path.join(localDir, subDir)
            if 'sub' in subDir:
                print('Files to download to ' + _localDir + ' : ' + str(len(lfnDict[subDir])) + ' file(s)')
            else:
                print(os.linesep.join(sorted(lfnDict[subDir])))

        msg = 'Do you want to download files:'
        UI().askRepeated(msg)
        
    msg = []
    msgDict = {'Successful':[], 'Failed':[], 'Skipped':[]}
    withDuplicatedJobID = []
    for subDir in sorted(lfnDict):
        # filter duplicated jobIDs LFNs
        res = filterDuplicatedJobID(lfnDict[subDir])
        if res['OK'] and res['Value']:
            withDuplicatedJobID = res['Value'][1]
            lfnDict[subDir] = res['Value'][0]
        _localDir = os.path.join(localDir, subDir)
        if args.se:
            se = args.se
        else:
            se = None
        result = manager.get(sorted(lfnDict[subDir]), _localDir, args.long, validatingType, se=se)
        if not result['OK']:
            msg.append('\nSuccessfully downloaded files:')
            msg.extend(msgDict['Successful'])
            msg.append('\nFailed files:')
            msg.extend(msgDict['Failed'])
            msg.append('')
            msg.extend(msgDict['Skipped'])
            msg.append('\n')            
            msg.append('Error %s happened' % result['Message'])
            return '\n'.join(msg)

        files = [os.path.join(_localDir, x) for x in result['Value']['Successful']]
        files = os.linesep.join(files)
        if len(files):
            msgDict['Successful'].append('%s in %s\n' % (files, _localDir))

        files = [os.path.join(_localDir, x) for x in result['Value']['Failed']]
        files = os.linesep.join(files)
        if len(files):
            msgDict['Failed'].append('%s in %s\n' % (files, _localDir))
            
            if args.failed_lfns:
                store_failed_lfns = open(args.failed_lfns, "a+")
                for failed_file in result['Value']['Failed']:
                    if failed_file not in store_failed_lfns.read():
                        store_failed_lfns.write("%s\n" % failed_file)
                store_failed_lfns.close()
                msgDict['Failed'].append('\nThe list of failed files is stored in "%s"' % (args.failed_lfns)) 

        if len(result['Value']['Skipped']):
            msgDict['Skipped'].append('Skip %d existing files in %s' % (len(result['Value']['Skipped']), _localDir))
        if withDuplicatedJobID:
            msgDict['Skipped'].append('\nFiles with duplicated jobID, not downloaded:')
            msgDict['Skipped'].extend(withDuplicatedJobID)
            msgDict['Skipped'].append('(See https://confluence.desy.de/display/BI/GBasf2+FAQ#GBasf2FAQ-OutputfileswithduplicatedJobID)\n')
    msg.append('\nSuccessfully downloaded files:')
    msg.extend(msgDict['Successful'])
    msg.append('\nFailed files:')
    msg.extend(msgDict['Failed'])
    msg.append('')
    msg.extend(msgDict['Skipped'])
    print('\n'.join(msg))

    _localDir_ = os.path.join(localDir, args.dataset)
    _subDir_ = [os.path.join(_localDir_, subDir) for subDir in os.listdir(_localDir_) if 'sub' in subDir]

    localFiles = [[os.path.join(subDir, file) for file in os.listdir(subDir)] for subDir in _subDir_]
    localFiles = [file for sublist in localFiles for file in sublist]

    formatFiles = set(file.split('.')[-1] for file in localFiles)
    for format in formatFiles:
        files = [file for file in localFiles if file.endswith('.' + format)]
        res = filterDuplicatedJobID(files)
        withDuplicatedJobID_local = res['Value'][1]
        if withDuplicatedJobID_local:
            print('\nFiles with duplicated jobID in the local directory:')
            print(*withDuplicatedJobID_local, sep='\n')
            action = 'deleted'
            def removing_localDuplicatedJobIDs():
                for job in withDuplicatedJobID_local:
                    os.remove(job)
            if args.force:
                removing_localDuplicatedJobIDs()
            else:
                question = '\nDo you want to delete jobs with duplicated JobID found locally?:'
                res = UI().yes_no_dialog(question)
                if not res['OK']:
                    return S_ERROR("Deleting process for local duplicated jobs is failed.")
                answer = res['Value']
                if answer: removing_localDuplicatedJobIDs()
                else: action = 'kept'
            print('\nFiles with duplicated jobID were %s locally.' % action)
    return ''

