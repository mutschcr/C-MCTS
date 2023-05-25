#include "safegridworld.h"
#include "utils.h"

using namespace std;
using namespace UTILS;

SAFEGRIDWORLD::SAFEGRIDWORLD(int size, int n_traps, double discount,std::string type)
: Grid(size, size), size(size),num_traps(n_traps)
{
NumActions = 8;
NumObservations = 1;
RewardRange = 80;
env_type = type;
Discount = discount;
StartPos = COORD(0,0);
Goal = COORD(size-1, size-1);
Grid.SetAllValues(0);
if ((size==8) && (num_traps==7))
{
COORD coord_traps[] = {COORD(2,5),COORD(3,5),COORD(4,5),COORD(5,5),
                    COORD(5,4),COORD(5,3),COORD(5,2)};
for (int i=0;i<num_traps;i++)
{
Grid(coord_traps[i])=i+1;
trap_pos.push_back(coord_traps[i]);
}
}
else if ((size==4) && (num_traps==4))
{
    COORD coord_traps[] = {COORD(1,2),COORD(2,2),COORD(2,1),COORD(1,1)};
    for (int i=0;i<num_traps;i++)
{
Grid(coord_traps[i])=i+1;
trap_pos.push_back(coord_traps[i]);
}
}
current_state = CreateStartState();
}
void SAFEGRIDWORLD::resetStartPos()
{
StartPos = COORD(Random(size/2-1),Random(size/2-1));
    if (Grid(StartPos)>0)
    {
        StartPos.X -=1 ; StartPos.Y -=1;
    }
current_state = CreateStartState();
}
COORD SAFEGRIDWORLD::getStartPos()
{
    return StartPos;
}
void SAFEGRIDWORLD::setStartPos(COORD iPos)
{
StartPos = iPos;
}
STATE* SAFEGRIDWORLD::CreateStartState() const
{
    SAFEGRIDWORLD_STATE* gridstate = MemoryPool.Allocate();
    gridstate->AgentPos = StartPos;
    gridstate->terminal = 0;
    return gridstate;
}


void SAFEGRIDWORLD::FreeState(STATE* state) const
{
    SAFEGRIDWORLD_STATE* gridstate = safe_cast<SAFEGRIDWORLD_STATE*>(state);
    MemoryPool.Free(gridstate);
}

py::array_t<int> SAFEGRIDWORLD::get_state() {

  SAFEGRIDWORLD_STATE& gridstate = safe_cast<SAFEGRIDWORLD_STATE&>(*current_state);

  /*  allocate the buffer */
  int n = 2+1;
  py::array_t<int> state = py::array_t<int>(n);
  py::buffer_info buf_state = state.request();

  int *ptr1 = (int *) buf_state.ptr;
   ptr1[0] = gridstate.AgentPos.X;
   ptr1[1] = gridstate.AgentPos.Y;
   ptr1[2] = gridstate.terminal;
  
  return state;
}
void SAFEGRIDWORLD::set_state(py::array_t<int> istate){

    SAFEGRIDWORLD_STATE& gridstate = safe_cast<SAFEGRIDWORLD_STATE&>(*current_state);
py::buffer_info buf = istate.request();

int n = 2 + 1;
  int *ptr1 = (int *) buf.ptr;
   gridstate.AgentPos.X=ptr1[0];
   gridstate.AgentPos.Y=ptr1[1];
   gridstate.terminal=ptr1[2];

}
int SAFEGRIDWORLD::Step_cpp(int action, RC& rewardcost) const
{
SAFEGRIDWORLD_STATE& gridstate = safe_cast<SAFEGRIDWORLD_STATE&>(*current_state);
rewardcost.R = 0;
rewardcost.C = 0;
int observation = 0;  
gridstate.terminal = 0;
if ((env_type.compare("lofi")!=0) & (gridstate.AgentPos.X>=2) & (gridstate.AgentPos.X<=5) & (gridstate.AgentPos.Y>=6) & (gridstate.AgentPos.Y<=7))
{
    double p = RandomDouble(0,1);
    if ((p > 0.8) & (env_type.compare("real")==0))
    {
       action=1; // head south
    }
    else if ((p > 0.75) & (env_type.compare("hifi")==0))
    {
        action = 1; // head south
    }
}
switch (action)
{
    case 0: // turn east
        if (gridstate.AgentPos.X + 1 < size)
            { gridstate.AgentPos.X++; break;}
        else
            {rewardcost.R = -1000; gridstate.terminal = 1;}
    case 1: // turn south
        if (gridstate.AgentPos.Y-1>=0)
          {gridstate.AgentPos.Y--; break;}
        else    
            {rewardcost.R = -1000; gridstate.terminal = 1;}
    case 2: // turn west
        if (gridstate.AgentPos.X -1 >=0)
            {gridstate.AgentPos.X--; break;}
        else    
            {rewardcost.R=-1000; gridstate.terminal = 1;}
    case 3: // turn north
        if (gridstate.AgentPos.Y+1< size)
            {gridstate.AgentPos.Y++;  break;}
        else
            {rewardcost.R=-1000; gridstate.terminal = 1;}
    case 4: // turn north-east
        if ((gridstate.AgentPos.X+1 < size) && (gridstate.AgentPos.Y+1<size))
            {gridstate.AgentPos.X++ ; gridstate.AgentPos.Y++; break;}
        else
            {rewardcost.R=-1000; gridstate.terminal = 1;}
    case 5: // turn south-east
        if ((gridstate.AgentPos.X+1<size)&&(gridstate.AgentPos.Y-1>=0))
            {gridstate.AgentPos.X++; gridstate.AgentPos.Y--; break;}
        else
            {rewardcost.R=-1000; gridstate.terminal = 1;}
    case 6: // turn south-west
        if ((gridstate.AgentPos.X-1>=0)&&(gridstate.AgentPos.Y-1>=0))
            {gridstate.AgentPos.X--;gridstate.AgentPos.Y--;break;}
        else
            {rewardcost.R=-1000; gridstate.terminal = 1;}
    case 7: // turn north-west
        if ((gridstate.AgentPos.X-1>=0)&&(gridstate.AgentPos.Y+1<size))
            {gridstate.AgentPos.X--;gridstate.AgentPos.Y++;break;}
        else
            {rewardcost.R=-1000; gridstate.terminal = 1;}

}           
    if (Grid(gridstate.AgentPos) > 0 ) // new position is a trap
        {rewardcost.C = 1;rewardcost.R -=1;}
    else if (gridstate.AgentPos==Goal) 
        {rewardcost.R = 100; gridstate.terminal = 1;}
    else
        rewardcost.R -=1; // penalty of -1 for staying longer

    return gridstate.terminal; 
}
py::array_t<double> SAFEGRIDWORLD::Step(int action) const
{   
    RC rewardcost(0.0,0.0);
    int terminal = Step_cpp(action,rewardcost);
    py::array_t<double> RC_numpy = py::array_t<double>(2);
    py::buffer_info buf_RC = RC_numpy.request();
    double *ptr1 = (double *) buf_RC.ptr;
    ptr1[0] = rewardcost.R;
    ptr1[1] = rewardcost.C;
    return RC_numpy;
}
py::array_t<double> SAFEGRIDWORLD::Rollout(int treeDepth)
{

    py::array_t<double> RC_numpy = py::array_t<double>(2);
    py::buffer_info buf_RC = RC_numpy.request();
    double *ptr1 = (double *) buf_RC.ptr;
    RC totalRewardCost(0.0, 0.0);
    int n_rollouts = 10;
    int x, y, t;
    SAFEGRIDWORLD_STATE& gridstate = safe_cast<SAFEGRIDWORLD_STATE&>(*current_state);
    x = gridstate.AgentPos.X;
    y = gridstate.AgentPos.Y;
    t = gridstate.terminal;
    for (int i=0;i<n_rollouts;i++)
    {
        double discount = 1.0;
        int terminal = 0;
        gridstate.AgentPos.X = x;
        gridstate.AgentPos.Y = y;
        gridstate.terminal = t;
            for(int numSteps = 0;numSteps < 132-treeDepth && !terminal;numSteps++)
            {   
                RC rewardcost;
                vector<int> preferred = getpreferredvecactions();
                int action = preferred[Random(preferred.size())];
                terminal = Step_cpp(action,rewardcost);
                totalRewardCost += rewardcost * discount;
                discount *= Discount;
            }
    }
    totalRewardCost = totalRewardCost/n_rollouts;
    ptr1[0] = totalRewardCost.R;
    ptr1[1] = totalRewardCost.C;
    return RC_numpy;

}
STATE* SAFEGRIDWORLD::Copy(const STATE& state) const
{
    const SAFEGRIDWORLD_STATE& gridstate = safe_cast<const SAFEGRIDWORLD_STATE&>(state);
    SAFEGRIDWORLD_STATE* newstate = MemoryPool.Allocate();
    *newstate = gridstate;
    return newstate;
}

vector<int> SAFEGRIDWORLD::getlegalvecactions() const
{

    const SAFEGRIDWORLD_STATE& gridstate =
        safe_cast<const SAFEGRIDWORLD_STATE&>(*current_state);
    vector<int> legal;
        if (gridstate.AgentPos.X + 1 < size)
             legal.push_back(COORD::E_EAST);

        if (gridstate.AgentPos.Y-1>=0)
           legal.push_back(COORD::E_SOUTH);
    
        if (gridstate.AgentPos.X -1 >=0)
            legal.push_back(COORD::E_WEST);

        if (gridstate.AgentPos.Y+1< size)
            legal.push_back(COORD::E_NORTH);
       
        if ((gridstate.AgentPos.X+1 < size) && (gridstate.AgentPos.Y+1<size))
            legal.push_back(COORD::E_NORTHEAST);
    
        if ((gridstate.AgentPos.X+1<size)&&(gridstate.AgentPos.Y-1>=0))
            legal.push_back(COORD::E_SOUTHEAST);
       
        if ((gridstate.AgentPos.X-1>=0)&&(gridstate.AgentPos.Y-1>=0))
             legal.push_back(COORD::E_SOUTHWEST);
        
        if ((gridstate.AgentPos.X-1>=0)&&(gridstate.AgentPos.Y+1<size))
            legal.push_back(COORD::E_NORTHWEST);
    
    return legal;
}

vector<int> SAFEGRIDWORLD::getpreferredvecactions() const
{

	const SAFEGRIDWORLD_STATE& gridstate =
        safe_cast<const SAFEGRIDWORLD_STATE&>(*current_state);
    vector<int> preferred;
        if (gridstate.AgentPos.X + 1 < size)
             preferred.push_back(COORD::E_EAST);

        if (gridstate.AgentPos.Y+1< size)
            preferred.push_back(COORD::E_NORTH);
       
        if ((gridstate.AgentPos.X+1 < size) && (gridstate.AgentPos.Y+1<size))
            preferred.push_back(COORD::E_NORTHEAST);
        
    return preferred;
 
}
void SAFEGRIDWORLD::GeneratePreferred(const STATE& state,
    vector<int>& actions, const STATUS& status) const
{
	const SAFEGRIDWORLD_STATE& gridstate =
        safe_cast<const SAFEGRIDWORLD_STATE&>(state);
    
        if (gridstate.AgentPos.X + 1 < size)
             actions.push_back(COORD::E_EAST);

        if (gridstate.AgentPos.Y+1< size)
            actions.push_back(COORD::E_NORTH);
       
        if ((gridstate.AgentPos.X+1 < size) && (gridstate.AgentPos.Y+1<size))
            actions.push_back(COORD::E_NORTHEAST);
 
}
py::array_t<int> SAFEGRIDWORLD::GenerateLegal() const
{
    vector<int> legal = getlegalvecactions(); 
    
      /*  allocate the buffer */
    int n = legal.size();
    py::array_t<int> actions = py::array_t<int>(n);
    py::buffer_info buf_actions = actions.request();
    int *ptr1 = (int *)buf_actions.ptr;

    for (int i=0;i<n;i++)
    {
        ptr1[i] = legal[i];
    }

    return actions;

}
void SAFEGRIDWORLD::DisplayState(const STATE& state, std::ostream& ostr) const
{
    const SAFEGRIDWORLD_STATE& gridstate = safe_cast<const SAFEGRIDWORLD_STATE&>(state);
    ostr << endl;
    for (int x = 0; x < size + 2; x++)
        ostr << "- ";
    ostr << endl;
    for (int y = size - 1; y >= 0; y--)
    {
        ostr << "| ";
        for (int x = 0; x < size; x++)
        {
            COORD pos(x, y);
            int val = Grid(pos);
            if (gridstate.AgentPos == COORD(x, y))
                ostr << "A ";
            else if (val > 0)
                ostr << "# ";
            else if (Goal == COORD(x,y))
                ostr << "G ";
            else
                ostr << ". ";
        }
        ostr << "|" << endl;
    }
    for (int x = 0; x < size + 2; x++)
        ostr << "- ";
    ostr << endl;
}