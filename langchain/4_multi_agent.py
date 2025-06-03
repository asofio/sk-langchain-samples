import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

class AgentType(Enum):
    """Types of agents in the travel planning system."""
    TRAVEL_PLANNER = "travel_planner"
    BUDGET_ANALYST = "budget_analyst"
    ACTIVITY_SPECIALIST = "activity_specialist"
    BOOKING_COORDINATOR = "booking_coordinator"

@dataclass
class TravelRequest:
    """Data structure for travel requests."""
    destination: str
    travel_dates: str
    budget: float
    travelers: int
    interests: List[str]
    accommodation_type: str = "hotel"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class AgentResponse:
    """Data structure for agent responses."""
    agent_type: AgentType
    message: str
    data: Dict[str, Any]
    next_agent: Optional[AgentType] = None
    handoff_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_type": self.agent_type.value,
            "message": self.message,
            "data": self.data,
            "next_agent": self.next_agent.value if self.next_agent else None,
            "handoff_reason": self.handoff_reason
        }

class BaseAgent:
    """Base class for all travel planning agents."""
    
    def __init__(self, agent_type: AgentType, system_prompt: str):
        """Initialize the agent with Azure OpenAI model and system prompt."""
        try:
            self.agent_type = agent_type
            self.llm = AzureChatOpenAI(
                azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
                azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT_NAME"],
                api_version=os.environ["AZURE_OPENAI_API_VERSION"],
                temperature=0.7,  # Creative but consistent
            )
            
            # Create the prompt template
            self.prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("user", "{input}")
            ])
            
            # Create the processing chain
            self.chain = self.prompt | self.llm | StrOutputParser()
            
        except KeyError as e:
            raise ValueError(f"Missing required environment variable: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize {agent_type.value} agent: {e}")
    
    def process(self, input_data: str) -> AgentResponse:
        """Process input and return agent response."""
        raise NotImplementedError("Subclasses must implement process method")

class TravelPlannerAgent(BaseAgent):
    """Agent responsible for initial travel planning and coordination."""
    
    def __init__(self):
        system_prompt = """You are a Senior Travel Planner specializing in creating comprehensive travel itineraries.
        
        Your responsibilities:
        1. Analyze travel requests and validate requirements
        2. Create high-level travel plans with destinations and timing
        3. Identify budget constraints and hand off to Budget Analyst
        4. Coordinate with other agents to build complete itineraries
        
        When you receive a travel request, create a preliminary itinerary and determine:
        - If budget analysis is needed (hand off to budget_analyst)
        - If activity planning is needed (hand off to activity_specialist)
        - If booking coordination is needed (hand off to booking_coordinator)
        
        Always provide detailed reasoning for your decisions and handoff recommendations.
        Respond in a professional but friendly tone."""
        
        super().__init__(AgentType.TRAVEL_PLANNER, system_prompt)
    
    def process(self, input_data: str) -> AgentResponse:
        """Process travel request and create initial plan."""
        try:
            response = self.chain.invoke({"input": input_data})
            
            # Analyze if budget analysis is needed
            needs_budget_analysis = "budget" in input_data.lower() or "$" in input_data
            
            # Create response data
            data = {
                "preliminary_plan": response,
                "timestamp": datetime.now().isoformat(),
                "analysis_needed": ["budget", "activities", "booking"]
            }
            
            next_agent = AgentType.BUDGET_ANALYST if needs_budget_analysis else AgentType.ACTIVITY_SPECIALIST
            handoff_reason = "Budget analysis required" if needs_budget_analysis else "Activity planning required"
            
            return AgentResponse(
                agent_type=self.agent_type,
                message=f"🗺️ Travel Planner: {response}",
                data=data,
                next_agent=next_agent,
                handoff_reason=handoff_reason
            )
            
        except Exception as e:
            return AgentResponse(
                agent_type=self.agent_type,
                message=f"❌ Error in travel planning: {e}",
                data={"error": str(e)},
                next_agent=None
            )

class BudgetAnalystAgent(BaseAgent):
    """Agent responsible for budget analysis and cost optimization."""
    
    def __init__(self):
        system_prompt = """You are a Budget Analysis Specialist for travel planning.
        
        Your responsibilities:
        1. Analyze travel budgets and provide cost breakdowns
        2. Suggest budget optimizations and alternatives
        3. Calculate estimated costs for accommodations, flights, activities, and meals
        4. Provide budget-friendly alternatives when requested
        
        When analyzing budgets:
        - Break down costs by category (flights, accommodation, food, activities, transportation)
        - Suggest cost-saving tips and alternatives
        - Flag potential budget issues or overruns
        - Recommend budget adjustments if necessary
        
        After budget analysis, hand off to Activity Specialist for activity planning.
        Be precise with numbers and provide realistic cost estimates."""
        
        super().__init__(AgentType.BUDGET_ANALYST, system_prompt)
    
    def process(self, input_data: str) -> AgentResponse:
        """Analyze budget and provide cost breakdown."""
        try:
            response = self.chain.invoke({"input": input_data})
            
            # Create budget analysis data
            data = {
                "budget_analysis": response,
                "cost_categories": ["flights", "accommodation", "food", "activities", "transportation"],
                "budget_status": "analyzed",
                "timestamp": datetime.now().isoformat()
            }
            
            return AgentResponse(
                agent_type=self.agent_type,
                message=f"💰 Budget Analyst: {response}",
                data=data,
                next_agent=AgentType.ACTIVITY_SPECIALIST,
                handoff_reason="Budget analyzed, now planning activities within budget constraints"
            )
            
        except Exception as e:
            return AgentResponse(
                agent_type=self.agent_type,
                message=f"❌ Error in budget analysis: {e}",
                data={"error": str(e)},
                next_agent=None
            )

class ActivitySpecialistAgent(BaseAgent):
    """Agent responsible for planning activities and experiences."""
    
    def __init__(self):
        system_prompt = """You are an Activity Planning Specialist with expertise in creating memorable travel experiences.
        
        Your responsibilities:
        1. Research and recommend activities based on traveler interests
        2. Create day-by-day activity schedules
        3. Consider budget constraints from previous analysis
        4. Suggest both popular attractions and hidden gems
        
        When planning activities:
        - Match activities to traveler interests and demographics
        - Consider timing, weather, and seasonal factors
        - Balance must-see attractions with unique local experiences
        - Provide alternatives for different energy levels and budgets
        - Include practical information (opening hours, booking requirements)
        
        After activity planning, hand off to Booking Coordinator for reservations.
        Be creative and focus on creating unique, memorable experiences."""
        
        super().__init__(AgentType.ACTIVITY_SPECIALIST, system_prompt)
    
    def process(self, input_data: str) -> AgentResponse:
        """Plan activities and experiences."""
        try:
            response = self.chain.invoke({"input": input_data})
            
            # Create activity planning data
            data = {
                "activity_plan": response,
                "activity_types": ["cultural", "adventure", "relaxation", "dining", "shopping"],
                "schedule_created": True,
                "timestamp": datetime.now().isoformat()
            }
            
            return AgentResponse(
                agent_type=self.agent_type,
                message=f"🎯 Activity Specialist: {response}",
                data=data,
                next_agent=AgentType.BOOKING_COORDINATOR,
                handoff_reason="Activities planned, ready for booking coordination"
            )
            
        except Exception as e:
            return AgentResponse(
                agent_type=self.agent_type,
                message=f"❌ Error in activity planning: {e}",
                data={"error": str(e)},
                next_agent=None
            )

class BookingCoordinatorAgent(BaseAgent):
    """Agent responsible for booking coordination and final arrangements."""
    
    def __init__(self):
        system_prompt = """You are a Booking Coordination Specialist responsible for finalizing travel arrangements.
        
        Your responsibilities:
        1. Coordinate all bookings based on previous agent recommendations
        2. Provide booking timeline and priority order
        3. Suggest booking platforms and methods
        4. Create final travel checklist and confirmations
        
        When coordinating bookings:
        - Prioritize time-sensitive bookings (flights, popular restaurants)
        - Provide specific booking instructions and links where possible
        - Create a timeline for when to make each booking
        - Include cancellation policies and travel insurance recommendations
        - Summarize the complete travel plan
        
        This is the final step in the travel planning process.
        Be thorough and provide actionable booking instructions."""
        
        super().__init__(AgentType.BOOKING_COORDINATOR, system_prompt)
    
    def process(self, input_data: str) -> AgentResponse:
        """Coordinate bookings and finalize arrangements."""
        try:
            response = self.chain.invoke({"input": input_data})
            
            # Create booking coordination data
            data = {
                "booking_plan": response,
                "booking_priority": ["flights", "accommodation", "activities", "restaurants"],
                "travel_ready": True,
                "timestamp": datetime.now().isoformat()
            }
            
            return AgentResponse(
                agent_type=self.agent_type,
                message=f"📋 Booking Coordinator: {response}",
                data=data,
                next_agent=None,  # Final agent in the chain
                handoff_reason="Travel planning complete"
            )
            
        except Exception as e:
            return AgentResponse(
                agent_type=self.agent_type,
                message=f"❌ Error in booking coordination: {e}",
                data={"error": str(e)},
                next_agent=None
            )

class TravelPlanningOrchestrator:
    """Orchestrates the multi-agent travel planning system."""
    
    def __init__(self):
        """Initialize all agents."""
        self.agents = {
            AgentType.TRAVEL_PLANNER: TravelPlannerAgent(),
            AgentType.BUDGET_ANALYST: BudgetAnalystAgent(),
            AgentType.ACTIVITY_SPECIALIST: ActivitySpecialistAgent(),
            AgentType.BOOKING_COORDINATOR: BookingCoordinatorAgent()
        }
        self.conversation_history: List[AgentResponse] = []
    
    def process_travel_request(self, travel_request: TravelRequest) -> List[AgentResponse]:
        """Process a travel request through the multi-agent system."""
        try:
            print("🚀 Starting Multi-Agent Travel Planning System...")
            print("="*60)
            
            # Convert travel request to input string
            request_str = self._format_travel_request(travel_request)
            print(f"📝 Travel Request:\n{request_str}\n")
            
            # Start with Travel Planner
            current_agent_type = AgentType.TRAVEL_PLANNER
            current_input = request_str
            responses = []
            
            # Process through agent chain with handoffs
            max_iterations = 10  # Prevent infinite loops
            iteration = 0
            
            while current_agent_type and iteration < max_iterations:
                iteration += 1
                print(f"🔄 Iteration {iteration}: Processing with {current_agent_type.value}")
                
                # Get current agent
                agent = self.agents[current_agent_type]
                
                # Build context from previous responses
                context = self._build_context(responses, current_input)
                
                # Process with current agent
                response = agent.process(context)
                responses.append(response)
                
                # Print agent response
                print(f"\n{response.message}\n")
                if response.handoff_reason:
                    print(f"🔀 Handoff Reason: {response.handoff_reason}")
                
                # Check for handoff to next agent
                if response.next_agent:
                    current_agent_type = response.next_agent
                    current_input = response.message  # Pass current response as input
                    print(f"➡️  Handing off to: {current_agent_type.value}\n")
                    print("-" * 40)
                else:
                    print("✅ Travel planning complete!")
                    break
            
            self.conversation_history.extend(responses)
            return responses
            
        except Exception as e:
            error_response = AgentResponse(
                agent_type=AgentType.TRAVEL_PLANNER,
                message=f"❌ System error: {e}",
                data={"error": str(e)},
                next_agent=None
            )
            return [error_response]
    
    def _format_travel_request(self, request: TravelRequest) -> str:
        """Format travel request for agent processing."""
        return f"""
        Travel Request Details:
        - Destination: {request.destination}
        - Travel Dates: {request.travel_dates}
        - Budget: ${request.budget:,.2f}
        - Number of Travelers: {request.travelers}
        - Interests: {', '.join(request.interests)}
        - Accommodation Type: {request.accommodation_type}
        """
    
    def _build_context(self, previous_responses: List[AgentResponse], current_input: str) -> str:
        """Build context from previous agent responses."""
        if not previous_responses:
            return current_input
        
        context_parts = [current_input]
        context_parts.append("\nPrevious Agent Analysis:")
        
        for response in previous_responses[-2:]:  # Include last 2 responses for context
            context_parts.append(f"\n{response.agent_type.value}: {response.message}")
        
        return "\n".join(context_parts)
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation."""
        return {
            "total_responses": len(self.conversation_history),
            "agents_involved": [r.agent_type.value for r in self.conversation_history],
            "final_status": "complete" if self.conversation_history else "not_started",
            "timestamp": datetime.now().isoformat()
        }
    
    def export_travel_plan(self, filename: Optional[str] = None) -> str:
        """Export the complete travel plan to a JSON file."""
        if not filename:
            filename = f"travel_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        export_data = {
            "travel_plan": [response.to_dict() for response in self.conversation_history],
            "summary": self.get_conversation_summary()
        }
        
        try:
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            print(f"📄 Travel plan exported to: {filename}")
            return filename
        except Exception as e:
            print(f"❌ Failed to export travel plan: {e}")
            return ""

def main():
    """Main function demonstrating the multi-agent travel planning system."""
    try:
        print("🌍 Multi-Agent Travel Planning System")
        print("="*60)
        
        # Initialize the orchestrator
        orchestrator = TravelPlanningOrchestrator()
        
        # Example travel requests
        travel_requests = [
            TravelRequest(
                destination="Tokyo, Japan",
                travel_dates="March 15-25, 2025",
                budget=4500.00,
                travelers=2,
                interests=["culture", "food", "technology", "temples"],
                accommodation_type="boutique hotel"
            ),
            TravelRequest(
                destination="Iceland",
                travel_dates="June 10-17, 2025",
                budget=3200.00,
                travelers=1,
                interests=["nature", "photography", "adventure", "hot springs"],
                accommodation_type="guesthouse"
            )
        ]
        
        # Process each travel request
        for i, request in enumerate(travel_requests, 1):
            print(f"\n{'='*20} TRAVEL REQUEST {i} {'='*20}")
            
            # Process the request through the multi-agent system
            responses = orchestrator.process_travel_request(request)
            
            # Export the travel plan
            export_file = f"travel_plan_{i}.json"
            orchestrator.export_travel_plan(export_file)
            
            # Print summary
            summary = orchestrator.get_conversation_summary()
            print(f"\n📊 Planning Summary:")
            print(f"   • Agents involved: {len(set(summary['agents_involved']))}")
            print(f"   • Total interactions: {summary['total_responses']}")
            print(f"   • Status: {summary['final_status']}")
            
            if i < len(travel_requests):
                input("\nPress Enter to continue to next travel request...")
                # Reset for next request
                orchestrator.conversation_history.clear()
        
        print("\n🎉 Multi-Agent Travel Planning Demo Complete!")
        print("\nKey Features Demonstrated:")
        print("• Agent handoffs based on expertise")
        print("• Context sharing between agents")
        print("• Specialized agent responsibilities")
        print("• Error handling and recovery")
        print("• Conversation export and logging")
        
    except ValueError as e:
        print(f"❌ Configuration Error: {e}")
        print("Please ensure your .env file contains:")
        print("- AZURE_OPENAI_ENDPOINT")
        print("- AZURE_OPENAI_DEPLOYMENT_NAME") 
        print("- AZURE_OPENAI_API_VERSION")
    except Exception as e:
        print(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()
