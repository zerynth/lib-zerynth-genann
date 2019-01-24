#include "zerynth.h"
#include "genann/genann.h"


C_NATIVE(_genann_create)
{
    NATIVE_UNWARN();
    int32_t ninputs,noutputs, nlayers, nhidden,size;

    *res = MAKE_NONE();

    if (parse_py_args("iiii", nargs, args, &ninputs, &noutputs, &nlayers, &nhidden) != 4)
        return ERR_TYPE_EXC;

    size = genann_size(ninputs, nlayers, nhidden, noutputs);
    if (!size)
        return ERR_VALUE_EXC;
    PBytes *pb = pbytes_new(size,NULL);
    uint8_t *ann = (uint8_t*)PSEQUENCE_BYTES(pb);
    genann_init(ninputs, nlayers, nhidden, noutputs,(genann*)ann);

    *res = (PObject*)pb;

    return ERR_OK;
}

C_NATIVE(_genann_run)
{
    NATIVE_UNWARN();
    PBytes *pb;
    PList *pl,*po;
    genann *ann;
    double *inputs;
    double *outputs;
    int i;

    *res = MAKE_NONE();
    if (nargs!=3)
        return ERR_TYPE_EXC;
    pb = (PBytes*) args[0];
    pl = (PList*) args[1];
    po = (PList*) args[2];

    //extract ann
    ann = (genann*)PSEQUENCE_BYTES(pb);
    //convert inputs to double[]
    inputs = (double*)gc_malloc(sizeof(double)*ann->inputs);

    //convert inputs
    for(i=0;i<PSEQUENCE_ELEMENTS(pl);i++){
        PFloat *f = (PFloat*) PLIST_ITEM(pl,i);
        double d = (double)FLOAT_VALUE(f);
        inputs[i] = d;
    }

    //run ann
    outputs = genann_run(ann,inputs);

    //free inputs
    gc_free(inputs);

    //convert outputs to PList[float]
    for(i=0;i<ann->outputs;i++){
        PFloat *pf = pfloat_new((float)outputs[i]);
        PLIST_SET_ITEM(po,i,pf);
    }

    *res = po;
    return ERR_OK;
}


C_NATIVE(_genann_set_weight)
{
    NATIVE_UNWARN();
    PBytes *pb;
    PFloat *w;
    PObject* i;
    int j;
    genann *ann;

    *res = MAKE_NONE();
    if (nargs!=3)
        return ERR_TYPE_EXC;
    pb = (PBytes*) args[0];
    i = args[1];
    w = (PFloat*) args[2];
    j = PSMALLINT_VALUE(i);

    //extract ann
    ann = (genann*)PSEQUENCE_BYTES(pb);
    double d = (double)FLOAT_VALUE(w);
    ann->weight[j]=d;

    return ERR_OK;
}


