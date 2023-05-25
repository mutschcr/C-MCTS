#ifndef NODE_H
#define NODE_H

#include "utils.h"
#include <iostream>

class SIMULATOR;
class QNODE;
class VNODE;
class STATE : public MEMORY_OBJECT
{
};

//-----------------------------------------------------------------------------

class RC
{
public:
    double R, C;
    RC() : R(0.0), C(0.0) {
    }
    RC(double R, double C) {
        this->R = R;
        this->C = C;
    }

    RC operator+(const RC &ref) const {
        return RC(R + ref.R, C + ref.C);
    }
    RC& operator+=(const RC &rhs) {
        this->R += rhs.R;
        this->C += rhs.C;
        return *this;
    }

    RC operator*(const double &value) const {
        return RC(R * value, C * value);
    }
    RC operator/(const double &value) const {
        return RC(R / value, C / value);
    }
};

inline RC operator* (const double &value, RC rhs) {
    return rhs * value;
}

template<class COUNT>
class VALUE
{
public:

    void Set(double count, RC value)
    {
        Count = count;
        Total = value * count;
    }

    void Add(RC totalReward)
    {
        Count += 1.0;
        Total += totalReward;
    }

    void Add(RC totalReward, COUNT weight)
    {
        Count += weight;
        Total += totalReward * weight;
    }

    RC GetValue() const
    {
        return Count == 0 ? Total : Total / Count;
    }

    COUNT GetCount() const
    {
        return Count;
    }

private:

    COUNT Count;
    RC Total;
};

//-----------------------------------------------------------------------------

class QNODE
{
public:

    VALUE<int> Value;
    VALUE<double> AMAF;

    void Initialise();

    VNODE*& Child(int c) { return Children[c]; }
    VNODE* Child(int c) const { return Children[c]; }

    void DisplayValue(std::ostream& ostr) const;
    void DisplayPolicy(double lambda, std::ostream& ostr) const;

    static int NumChildren;

private:

    std::vector<VNODE*> Children;

friend class VNODE;
};

//-----------------------------------------------------------------------------

class VNODE : public MEMORY_OBJECT
{
public:

    VALUE<int> Value;

    void Initialise();
    static VNODE* Create();
    static void Free(VNODE* vnode, const SIMULATOR& simulator);
    static void FreeAll();

    QNODE& Child(int c) { return Children[c]; }
    const QNODE& Child(int c) const { return Children[c]; }
    STATE * mystate;
    void SetChildren(int count, RC value);

    void DisplayValue( std::ostream& ostr) const;
    void DisplayPolicy(double lambda, std::ostream& ostr) const;

    static int NumChildren;

private:

    std::vector<QNODE> Children;
    static MEMORY_POOL<VNODE> VNodePool;
};

#endif // NODE_H
