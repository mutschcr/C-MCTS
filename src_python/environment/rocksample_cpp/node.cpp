#include "node.h"
#include "utils.h"
#include "rocksample.h"
using namespace std;
using namespace UTILS;

//-----------------------------------------------------------------------------

int QNODE::NumChildren = 0;

void QNODE::Initialise()
{
    assert(NumChildren);
    Children.resize(NumChildren);
    for (int observation = 0; observation < QNODE::NumChildren; observation++)
        Children[observation] = 0;
}

void QNODE::DisplayValue(ostream& ostr) const
{
    ostr << ": " << Value.GetValue().R << ", " << Value.GetValue().C << " (" << Value.GetCount() << ")\n";

    for (int observation = 0; observation < NumChildren; observation++)
    {
        if (Children[observation])
        {
            Children[observation]->DisplayValue(ostr);
        }
    }
}

void QNODE::DisplayPolicy(double lambda, ostream& ostr) const
{
    ostr << ": R=" << Value.GetValue().R << ", C=" << Value.GetValue().C
            << ", H=" << Value.GetValue().R - lambda * Value.GetValue().C
            << " (" << Value.GetCount() << ")\n";


    for (int observation = 0; observation < NumChildren; observation++)
    {
        if (Children[observation])
        {
            Children[observation]->DisplayPolicy(lambda, ostr);
        }
    }
}

//-----------------------------------------------------------------------------

MEMORY_POOL<VNODE> VNODE::VNodePool;

int VNODE::NumChildren = 0;

void VNODE::Initialise()
{
    assert(NumChildren);
    Children.resize(VNODE::NumChildren);
    for (int action = 0; action < VNODE::NumChildren; action++)
        Children[action].Initialise();
}

VNODE* VNODE::Create()
{
    VNODE* vnode = VNodePool.Allocate();
    vnode->Initialise();
    return vnode;
}

void VNODE::Free(VNODE* vnode, const SIMULATOR& simulator)
{
    simulator.FreeState(vnode->mystate); //NOTE: need to check if this is required
    VNodePool.Free(vnode);
    for (int action = 0; action < VNODE::NumChildren; action++)
        for (int observation = 0; observation < QNODE::NumChildren; observation++)
            if (vnode->Child(action).Child(observation))
                Free(vnode->Child(action).Child(observation), simulator);
}

void VNODE::FreeAll()
{
	VNodePool.DeleteAll();
}

void VNODE::SetChildren(int count, RC value)
{
    for (int action = 0; action < NumChildren; action++)
    {
        QNODE& qnode = Children[action];
        qnode.Value.Set(count, value);
        qnode.AMAF.Set(count, value);
    }
}

void VNODE::DisplayValue(ostream& ostr) const
{
    for (int action = 0; action < NumChildren; action++)
    {
        Children[action].DisplayValue(ostr);
    }
}

void VNODE::DisplayPolicy(double lambda, ostream& ostr) const
{

    double bestq = -Infinity;
    int besta = -1;
    for (int action = 0; action < NumChildren; action++)
    {
        double scalarizedValue = Children[action].Value.GetValue().R - lambda * Children[action].Value.GetValue().C;
        if (scalarizedValue > bestq)
        {
            besta = action;
            bestq = scalarizedValue;
        }
    }

    if (besta != -1)
    {
        Children[besta].DisplayPolicy(lambda, ostr);
    }
}

//-----------------------------------------------------------------------------
