@userCreds
def queryExpansion(args, type):
    manager = Manager()
    lfns = []
    for ds in args.dataset:
        result = manager.getLfns(args.subcate, ds, args.user, noFiles=True)
        if result['OK']:
            lfns.extend(result['Value'])

    attributes=[]
    for ds in lfns:
        if type == 'dataset':
            if os.path.abspath(ds):
                metaVersion = manager.getMetaVersion(ds)
                ds = manager.getDSMetaLPN(ds, metaVersion)
        elif type == 'datablock':
            if os.path.abspath(ds):
                metaVersion = manager.getMetaVersion(ds)
                ds = manager.getDMBMetaLPN(ds, metaVersion)                
        res = query(args, ds, type)
        if type == 'file':
            if args.output_csv and res:
                attributes.extend(res)

    if type == 'file' and attributes:
        nameColumns = [i[0] for i in attributes[0]]
        valueRows = [[i[1] for i in atb] for atb in attributes]
        with open(args.output_csv, 'w') as outcsv:
            metawriter = csv.writer(outcsv)
            metawriter.writerow(nameColumns)
            for value in valueRows:
                metawriter.writerow(value)
        return "Attributes of file(s) saved in " + args.output_csv
    return ''
