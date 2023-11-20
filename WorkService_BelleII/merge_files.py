def mergeInput(args):

    manager = Manager()
    lfns = manager.getLfns(args.subcate, args.dataset, args.user, False, noDirs=True, resolveDatablocks=(not args.noSubDir))
    def generate_list_to_merge(input_lfns, maxSize=50):

        """
        params:
            inputfiles: type=list, contains paths of input files to be merged.
            maxSize; type=float, per outputfile measured in [MB].
        """
        maxSize*=10**6
        sizes = [manager.getFileSize(lfn)['Value'] for lfn in input_lfns]
        input_lfns_to_merge = []
        if sum(sizes) > maxSize:
            ind_output = []
            cumulative = 0
            for lfn, size in zip(input_lfns, sizes):
                if cumulative <= maxSize:
                    ind_output.append(file)
                    cumulative += size
                    if lfn == input_lfns[-1]:
                        input_lfns_to_merge.append(ind_output)
                else:
                    input_lfns_to_merge.append(ind_output)
                    ind_output = [lfn]
                    cumulative = 0
        else:
            input_lfns_to_merge = [input_lfns]
        return input_lfns_to_merge

    def grouping_lfns(input_lfns):

        def name_and_format(lfn):
            filename = lfn.split("/")[-1]
            name = filename.split("_")[:-3]
            name = "_".join(name)
            fileformat = filename.split(".")[-1]
            return [name, fileformat]
        nonRepetead = []
        for lfn in input_lfns:
            duple = name_and_format(lfn)
            if duple not in nonRepetead:
                nonRepetead.append(duple)
        grouped = {}
        for duple in nonRepetead:
            ref = '-'.join(duple)
            grouped[ref] = []
            for lfn in input_lfns:
                if duple[0] in lfn and duple[1] in lfn:
                    grouped[ref].append(lfn)
        return grouped

    def merging(input_lfns, output_filename):
        args_merge = ["hadd", output_filename] + input_lfns
        hadd = subprocess.run(args_merge, capture_output=True)   
        if hadd.returncode != 0:
            sys.exit(hadd.returncode)

    grouped_lfns = grouping_lfns(lfns['Value'])
    for name_format, input_lfns in zip(grouped_lfns.keys(), grouped_lfns.values()):
        name = name_format.split('-')[0]
        fileformat = name_format.split('-')[1]

        input_lfns_to_merge = generate_list_to_merge(input_lfns)
        for count, packet in enumerate(input_lfns_to_merge):
            merging(packet, "{}_{}.{}".format(name, count, fileformat))

    return ''
