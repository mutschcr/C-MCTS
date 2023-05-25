#include "simulator.h"
#include "coord.h"
#include "grid.h"
#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
namespace py = pybind11;

class SAFEGRIDWORLD_STATE : public STATE
{
    public:
        COORD AgentPos;
        int terminal=0;

};

class SAFEGRIDWORLD : public SIMULATOR
{
public:   
    SAFEGRIDWORLD(int size, int traps, double discount=1.0,std::string type="lofi");
    virtual void FreeState(STATE* state) const;
    int Step_cpp(int action,RC& rewardcost) const;
    virtual py::array_t<double> Step(int action) const;
    std::vector<int> getlegalvecactions() const;
    std::vector<int> getpreferredvecactions() const;
    py::array_t<int> GenerateLegal() const;
    py::array_t<double> Rollout(int treeDepth);
 py::array_t<int> get_state();
     void set_state(py::array_t<int> istate);
        void GeneratePreferred(const STATE& state,
        std::vector<int>& legal, const STATUS& status) const;
    virtual STATE* CreateStartState() const;
    virtual STATE* Copy(const STATE& state) const;
    virtual void DisplayState(const STATE& state, std::ostream& ostr) const;
    void resetStartPos();
    COORD getStartPos();
    void setStartPos(COORD iPos);
    

private:
    GRID<int> Grid;
    std::vector<COORD> trap_pos;
    COORD StartPos,Goal;
    int size, num_traps;
    mutable MEMORY_POOL<SAFEGRIDWORLD_STATE> MemoryPool;
    STATE* current_state;
    
};
