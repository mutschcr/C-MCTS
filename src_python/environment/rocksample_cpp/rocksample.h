#ifndef ROCKSAMPLE_H
#define ROCKSAMPLE_H

#include "simulator.h"
#include "coord.h"
#include "grid.h"
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
namespace py = pybind11;

class ROCKSAMPLE_STATE : public STATE
{
public:

    COORD AgentPos;
    int terminal;
    struct ENTRY
    {
        bool Valuable;
        bool Collected;
        int Count;    				// Smart knowledge
        int Measured; 				// Smart knowledge
        double LikelihoodValuable;	// Smart knowledge
        double LikelihoodWorthless;	// Smart knowledge
        double ProbValuable;		// Smart knowledge
    };
    std::vector<ENTRY> Rocks;
    int Target; // Smart knowledge
};

class ROCKSAMPLE : public SIMULATOR
{
public:

    ROCKSAMPLE(int size, int rocks,double discount=1.0, bool isreal=true);
    virtual STATE* Copy(const STATE& state) const;
    virtual void Validate(const STATE& state) const;
    virtual STATE* CreateStartState() const;
    virtual void FreeState(STATE* state) const;
    int Step_cpp(int action,RC& rewardcost) const;
    virtual py::array_t<double> Step(int action) const;
    std::vector<int> getlegalvecactions() const;
    py::array_t<int> GenerateLegal() const;
    py::array_t<double> Rollout(int treeDepth);
    void GeneratePreferred(const STATE& state,
        std::vector<int>& legal, const STATUS& status) const;
    int GetAgentX();
    int GetAgentY();
    py::array_t<double> get_state();
    void set_state(py::array_t<double> istate);
    virtual bool LocalMove(STATE& state,
        int stepObservation, const STATUS& status) const;

    virtual void DisplayBeliefs(const BELIEF_STATE& beliefState,
        std::ostream& ostr) const;
    virtual void DisplayState(const STATE& state, std::ostream& ostr) const;
    virtual void DisplayObservation(const STATE& state, int observation, std::ostream& ostr) const;
    virtual void DisplayAction(int action, std::ostream& ostr) const;

protected:

    enum
    {
        E_NONE,
        E_GOOD,
        E_BAD
    };

    enum
    {
        E_SAMPLE = 4
    };

    void InitGeneral();
    void Init_5_5();
    void Init_5_7();
    void Init_7_8();
    void Init_11_11();
    int GetObservation(const ROCKSAMPLE_STATE& rockstate, int rock) const;
    int SelectTarget(const ROCKSAMPLE_STATE& rockstate) const;

    GRID<int> Grid;
    std::vector<COORD> RockPos;
    int Size, NumRocks;
    COORD StartPos;
    double HalfEfficiencyDistance;
    double SmartMoveProb;
    int UncertaintyCount;

private:

    mutable MEMORY_POOL<ROCKSAMPLE_STATE> MemoryPool;
    STATE* current_state;
};

#endif
