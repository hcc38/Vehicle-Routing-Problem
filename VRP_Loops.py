'''
This program solves the 'Vehicle Routing Problem' using Google's or-tools 
(https://developers.google.com/optimization/) and displays the service points
and route of each vehicle on OpenStreetMap using Folium. The vehicle routes
are cicular, i.e. start location = end location. Moreover, the start/end location
are the same for all vehicles. Relevant scenarios are, for example, logistics
to and from depot.) More complicated start and end location are possible
with or-tools. (See the or-tools manual: https://acrogenesis.com/or-tools/
documentation/user_manual/index.html)
All service points must be serviced or else error. (See the or-tools manual for
how to build in the flexibility.) The routing operation and the distance 
and duration between locations are obtained using OSRM API. These could
be modified easily to work with other API, such as Google Map API's. (Some
API's place limitation of the number of request allowed.)

Developed and operated using Spyder 2.3.8.

Input variables are:
    -coordinates of service locations
    -number of vehicles
    -capacity of vehicles (same for all, easily modified for dissimilar capacity)
    -demands at each service location
    -tune to encourage equal distance travelled or time (if proportional to distance)
      taken by each vehicle. Can be used to find the strategy with the fastest completion
      time (if time is proportional to distance). If (tuning) coefficient is set to zero, the solver
      attempts to search for strategy that minimises the total distance travelled by all vehicles.
    -set soft upper and lower bounds for the total route length or time (assuming time proportional
      to distance) for all vehicles and tune the strength of the conditions.
    -set how many vehicles are discouraged from servicing and tune the extent of this,
      e.g. if some vehicles are more expensive to operate.

Note: This is a working program that is under development and should increase in
sophistication over time. The program may also be used as a reference for working with
routing problems and the python version of Google's or-tools.

'''
##############################Import Modules###################################

import math
import json
from urllib2 import Request, urlopen, URLError
import Queue
import threading
#Folium 0.2.1
import folium
import numpy as np
import os
import pandas as pd
import time

#ortools 5.1.4045
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

######################Data & Set Values(User TODO)#############################

#A list of pick-up/location/node coordinates, i.e. [[Lat1,Long1], [Lat2,Long2],
#[Lat3,Long3],....]. locations[n] is given a node_index of n, an identity of that
#node. (Set your own location points.)

###Import from csv file. Replace string in ' ' with the csv file location and file
###name. Note that the first column has to be Latitude and the second Longitude.
##locations=pd.read_csv('C:\...\..\...\___.csv',names=['Lat.','Long.'], skiprows=0)

#Some 2015 Yellow Taxi Trip data (Pick up locations)
#Source: https://data.cityofnewyork.us/Transportation/2015-Yellow-Taxi-Trip-Data/ba8s-jw6u
locations=pd.DataFrame([[ 40.64485931, -73.78211212],
       [ 40.66588211, -73.80173492],
       [ 40.69569778, -73.98491669],
       [ 40.71608734, -74.00937653],
       [ 40.71871567, -73.97517395],
       [ 40.71909714, -73.98957062],
       [ 40.72087097, -73.99381256],
       [ 40.72202301, -74.00408936],
       [ 40.72467804, -73.99875641],
       [ 40.72569275, -73.99755859],
       [ 40.72710419, -73.99172974],
       [ 40.73020935, -74.00042725],
       [ 40.73022461, -73.99193573],
       [ 40.73054123, -73.98051453],
       [ 40.73109055, -73.98199463],
       [ 40.73907852, -73.98729706],
       [ 40.74047089, -73.89561462],
       [ 40.74289322, -73.97389984],
       [ 40.74406052, -73.99932861],
       [ 40.74414062, -73.98994446],
       [ 40.74612427, -73.97174835],
       [ 40.74770737, -73.99666595],
       [ 40.74973679, -73.97782898],
       [ 40.75123215, -73.9788208 ],
       [ 40.75203323, -74.00453186],
       [ 40.75555801, -73.96570587],
       [ 40.75592422, -73.97145081],
       [ 40.75598907, -73.96443176],
       [ 40.75900269, -73.97098541],
       [ 40.75912094, -73.98040009],
       [ 40.75930023, -73.97645569],
       [ 40.76028061, -73.965271  ],
       [ 40.7604332 , -73.98036194],
       [ 40.76086426, -73.96910095],
       [ 40.76106644, -73.98695374],
       [ 40.7611084 , -73.97316742],
       [ 40.76203156, -73.96604156],
       [ 40.76295853, -73.97379303],
       [ 40.76344681, -73.99279785],
       [ 40.7635498 , -73.97135925],
       [ 40.76362228, -73.95901489],
       [ 40.7638855 , -73.95417023],
       [ 40.76518631, -73.98399353],
       [ 40.76725769, -73.98439789],
       [ 40.7707901 , -73.96199036],
       [ 40.77116013, -73.98152924],
       [ 40.77159882, -73.98286438],
       [ 40.7716217 , -73.9503479 ],
       [ 40.77241898, -73.9466095 ],
       [ 40.77338791, -73.9552536 ],
       [ 40.77397919, -73.87302399],
       [ 40.77728271, -73.95211792],
       [ 40.77815628, -73.95637512],
       [ 40.77889252, -73.97786713],
       [ 40.77903366, -73.98126221],
       [ 40.78487015, -73.94933319],
       [ 40.79037094, -73.97373199],
       [ 40.81287003, -73.94177246]])
#Convert dataframe to numpy array (No need to touch!!!)
locations=locations.as_matrix()

#Number of locations/nodes/waypoints (No need to touch!!!)
num_locations = len(locations)

#Demands at each location, which could be anything e.g. number, volume, weight, etc.
#A list of demands; List length same as number of locations. Start and end point
#(called the depot) will have zero demand. demands[n] = demand at locations[n]. (Set your own.)
demands = [1]*(num_locations)

#Tries to enforce equi-distance or time (assuming time proportional to distance)
#travelled by each vehicle. (Set your own.)
GlobSpan = 5

#Capacity of each vehicle, which could be anything e.g. number, volume, weight, etc.
#(Set your own.)
vehicle_capacity = 9

#Number of vehicles available. (Set your own.)
num_vehicles = int(sum(demands)/vehicle_capacity) + 2

#Number of vehicles discouraged to pick-up (<=num_vehicles) and the discouragement
#strength. (Set your own.)
num_vehicles_discouraged = 0
vehicles_discouraged_coef = 0

#Try to avoid having total route length (in metres) or time (assuming time proportional to distance) >
#the value of 'SoftUpBound'. SoftUpBoundCoeff controls the strength of the condition.
SoftUpBound = 50000
SoftUpBoundCoeff = 0

#Try to avoid having total route length (in metres) or time (assuming time proportional to distance) <
#the value of 'SoftLwBound'. SoftLwBoundCoeff controls the strength of the condition.
SoftLwBound = 5000
SoftLwBoundCoeff = 0

#The start and end point of the routes. 'depot = n' means locations[n] is the depot.
#(Set your own.)
depot = 0

#Colours associated with the vehicles/routes, e.g. colour[0]=colour of vehicle number 0.
#More colours needed if number of vehicles exceeds 14 - just add more colours to list.
#Note: Some colours may not be good for the route polyline.
colour=['red', 'blue', 'green', 'purple', 'orange','darkgreen', 'darkred', 'black',\
 'darkblue', 'darkpurple', 'cadetblue', 'pink', 'lightblue','lightgreen']

#Time for route solver before abort (in ms). If no solution found, try tuning the 
#above parameters, in particular, the num_vehicles.
timelimit = 5000

#################Create Inputs Needed For VRP Solver#######################

#Class for communicating to OSRM API to get distance and duration between
#two locations/nodes, e.g. (x1,y1) & (x2,y2). Note: x's are longitude 
#(type float) & y's are latitude (type float). Can be changed easily to use Google
#API, e.g. just change the url_add to use Google's and Long./Lat. order 
#may be different, but request limit exists
class metadata(object):
    def __init__(self,x1,y1,x2,y2):
        origin_coor=str(y1)+','+str(x1)
        dest_coor=str(y2)+','+str(x2)
        url_add='http://router.project-osrm.org/route/v1/driving/%s;%s?overview=false' %(origin_coor,dest_coor)
        url_request = Request(url_add)
        url_response = urlopen(url_request)
        url_data = url_response.read()
        self.url_dict=json.loads(url_data)  #Convert JSON to dictionary type
    def distance(self): #Get distance from URL response
        return self.url_dict['routes'][0]['distance']
    def duration(self): #Get duration from URL response
        return self.url_dict['routes'][0]['duration']

#Class for generating a thread to get the distance between the two coordinates
#given in each element of the matrix 'matrix'. 'matrix' matrix is a dict object
#that looks like [ [..., [c1,c1],[c1,c2],...], [..., [c2,c1],[c2,c2],...], ....], 
#where cn=[Long, Lat].
class ThreadUrl(threading.Thread):
    def __init__(self, queue,distmat):
        threading.Thread.__init__(self)
        #'queue' contains a list of objects. In this case, an object is 
        #(from_node_thr, to_node_thr, x1, y1, x2, y2).See later for more clarity.
        self.queue = queue  

    def run(self):    
        while True:
            #Get an object from the queue and assign them to the variables on the
            #left of the equality. The object wil automatically be removed from
            #'queue'.
            from_node_thr, to_node_thr, x1, y1, x2, y2 = self.queue.get()
            #Create a 'metadata' class object.
            metadt=metadata(x1, y1, x2, y2)
            #'matrix'->'distmat': each element in the former, which contain two coordinates
            #is replaced with the distance between the 2 coordinates. The result is
            # the latter matrix.
            distmat[from_node_thr][to_node_thr] = metadt.distance()
            self.queue.task_done()

start_time = time.time()

#Number of waypoints/locations/nodes
size = len(locations)

#Initialising matrix 'distmat' that will contain the distances between all possible pair-wise
#combination of location. This is an empty matrix to be 'inserted' to the class
#ThreadUrl to collect the distances. 'TransitMat' is the equivalent for the duration.
distmat = {}
TransitMat={}
for from_node in xrange(size):
    distmat[from_node] = {}
    TransitMat[from_node] = {}
    for to_node in xrange(size):
        distmat[from_node][to_node]= {}
        TransitMat[from_node][to_node]= {}
        
#Create a matrix 'matrix' containing all possible pair-wise combination of location
#coordinates.
matrix = {}
for from_node in xrange(size):
    matrix[from_node] = {}
    for to_node in xrange(size):
          x1 = locations[from_node]
          x2 = locations[to_node]
          matrix[from_node][to_node] = [x1,x2]


#Create 'Queue' object
queue = Queue.Queue()
#Add each element of 'matrix' to the queue; position of element is tracked so that
#each element of the 'distmat' matrix corresponds to the distance between the
#coordinates combination appearing in the same position in 'matrix'. Note:node1 to
#node2 distance not necessarily = node2 to node1 distance for road networks.        
for from_node in xrange(size):
    for to_node in xrange(size):
        from_coor_x=matrix[from_node][to_node][0][0]
        from_coor_y=matrix[from_node][to_node][0][1]
        to_coor_x=matrix[from_node][to_node][1][0]
        to_coor_y=matrix[from_node][to_node][1][1]
        queue.put((from_node,to_node,from_coor_x,from_coor_y,to_coor_x,to_coor_y))

#Spawn 'size' numbers of threads. Pass to each thread the same 'Queue' instance
#and the same 'distmat' matrix. These two are not replicated in each thread but
#are shared between threads. 'distmat' matrix, an empty matrix, will be filled
#with distances.'size' is the number of locations but one can spawn other numbers of thread.
for i in range(size):
    t = ThreadUrl(queue,distmat)
    t.setDaemon(True)
    t.start()
    
#Wait on the queue until everything has been processed
queue.join()

print "Communicate with API (1)  "+ "--- %s seconds ---" % (time.time() - start_time)

#############################Set Up the VRP Solver###################################

#Create callback to calculate distances between points. A class defined in the
#Google developer website.
class CreateDistanceCallback(object):
    
  #Remember that 'distmat' is a matrix that contains the distances.    
  def __init__(self, distmat):
          self.matrix_CB = distmat
  
  #Give the distance between two nodes (the arguments). Extracts value from
  #'distmat' matrix.
  def Distance(self, from_node, to_node):
    return self.matrix_CB[from_node][to_node]

#Create callback to get demands at each location (as given in Google Developer Website).
class CreateDemandCallback(object):

  def __init__(self, demands):
    self.matrix = demands

  def Demand(self, from_node, to_node):
    return self.matrix[from_node]


def main():
    
  # Create routing model.
  if num_locations > 0:    
      
    routing = pywrapcp.RoutingModel(num_locations, num_vehicles, depot)
    search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()

    #The method for finding a first solution to the problem. To use other methods,
    #replace 'GLOBAL_CHEAPEST_ARC' with other methods, e.g. 'GLOBAL_CHEAPEST_ARC', etc.
    #Check out the command: dir(routing_enums_pb2.FirstSolutionStrategy)!
    search_parameters.first_solution_strategy = (routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    ###Check out the command: dir(routing_enums_pb2.LocalSearchMetaheuristic) for other methods!
    ##search_parameters.local_search_metaheuristic=(routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    #Time limit. Needed when metaheuristic is used. Also, stops the program if 
    #running for too long.
    search_parameters.time_limit_ms=timelimit
    
    #Put a callback to the distance.
    dist_callback = CreateDistanceCallback(distmat).Distance
    #Sets the cost function of the model such that the cost of a segment of a
    #route between node 'from' and 'to' is evaluator(from, to), whatever the
    #route or vehicle performing the route. (From: https://github.com/google/
    #or-tools/blob/master/src/constraint_solver/routing.h)
    routing.SetArcCostEvaluatorOfAllVehicles(dist_callback)

    #Put a callback to the demands.
    demands_callback = CreateDemandCallback(demands).Demand

    ##########Add a dimension/constraint. The 'demand' is this case.#############
    slack_max = 0 
    #Initial load in vehicles is set to zero, i.e. start capacity of a vehicle
    #= max capacity of a vehicle.
    fix_start_cumul_to_zero = True
    #A name given to the dimension/constraint.
    demand = "Demand"
    #Now, add the dimension into the problem.
    routing.AddDimension(demands_callback, slack_max, vehicle_capacity,fix_start_cumul_to_zero, demand)
    
    ##########Find max distance between depot and locations. (Not Used)##############
    #HorMat=pd.DataFrame.from_dict(distmat).as_matrix()
    #MaxDist=int(np.amax(HorMat[:,0])+np.amax(HorMat[0,:]))
    
    #######Add a dimension/constraint. The distance travelled by each vehicle is this case.########
    Horizon=1000000
    routing.AddDimension(dist_callback,Horizon,Horizon,True,"DistConst")
    
    #Call the dimension/constraint named "DistConst".
    DisConst = routing.GetDimensionOrDie("DistConst")

    #Reduce timespan (assuming time proportional to distance) to complete all jobs. Can
    #be changed to actual time easily, i.e. by creating a time travelled equivalent
    #of 'dist_callback' and use that in place of 'dist_callback' in
    #routing.AddDimension(dist_callback,Horizon,Horizon,True,"DistConst") above.
    #From https://github.com/google/or-tools/blob/master/src/constraint_solver/routing.h,
    #sets a cost proportional to the *global* dimension span, that is the
    #difference between the largest value of route end cumul variables and
    #the smallest value of route start cumul variables. In other words:
    #global_span_cost = coefficient * (Max(dimension end value) - Min(dimension start value))
    DisConst.SetGlobalSpanCostCoefficient(GlobSpan)
    
    ###Not used:Set the maximum total route length or time (assuming time proportional to distance)
    ###for 'vehicle_nbr' to MaxDist.
    ##DisConst.SetSpanUpperBoundForVehicle(MaxDist,vehicle_nbr)
    
    
    for vehicle_nbr in range(num_vehicles):
        #Penalise 'vehicle_nbr' if total route length or time (assuming time proportional
        #to distance) > SoftUpBound; Last argument is the penalty coefficient.
        DisConst.SetEndCumulVarSoftUpperBound(vehicle_nbr,SoftUpBound,SoftUpBoundCoeff)
        #Penalise 'vehicle_nbr' if total route length or time (assuming time proportional
        #to distance) < SoftLwBound; Last argument is the penalty coefficient.
        DisConst.SetEndCumulVarSoftLowerBound(vehicle_nbr,SoftLwBound,SoftLwBoundCoeff)

    #Discourage use of 'num_vehicles_discouraged' numbers of vehicles and set the how much
    #they are discouraged with 'vehicles_discouraged_coef'.
    for vehicle_nbr in range(num_vehicles):
        if vehicle_nbr in range(num_vehicles_discouraged): 
            DisConst.SetSpanCostCoefficientForVehicle(int(vehicles_discouraged_coef),vehicle_nbr)

    ##############################Solve VRP###########################################
    assignment = routing.SolveWithParameters(search_parameters)
    
    ###########################Display Result########################################
    ##################(References Google Developer Website.)#########################
    if assignment:
      print "Objective Value: " + str(assignment.ObjectiveValue()) + "\n"

      vehicle={}    
      for vehicle_nbr in range(num_vehicles):
        #First node of vehicle 'vehicle_nbr'; Equivalently, the depot.
        index = routing.Start(vehicle_nbr)
        #Second node of vehicle 'vehicle_nbr'.
        index_next = assignment.Value(routing.NextVar(index))
        route = ''
        route_dist = 0
        route_demand = 0
        vehicle[vehicle_nbr]=[]
        
        while not routing.IsEnd(index_next):
          #Get the 'node_index'. 'index' may not be equal to the index assigned
          #to the node at the beginning. Remember that 'locations[n] is given a node_index of n'.
          #'index' is used internally by the solver to solve the routing problem.
          node_index = routing.IndexToNode(index)
          node_index_next = routing.IndexToNode(index_next)
          vehicle[vehicle_nbr].extend([node_index])
          route += str(node_index) + " -> "
          # Add the distance to the next node.
          route_dist += dist_callback(node_index, node_index_next)
          # Add demand.
          route_demand += demands[node_index_next]
          index = index_next
          index_next = assignment.Value(routing.NextVar(index))

        node_index = routing.IndexToNode(index)
        node_index_next = routing.IndexToNode(index_next)
        vehicle[vehicle_nbr].extend([node_index])
        vehicle[vehicle_nbr].extend([node_index_next])
        route += str(node_index) + " -> " + str(node_index_next)
        route_dist += dist_callback(node_index, node_index_next)
        print "Route for vehicle " + str(vehicle_nbr) + ":\n\n" + route + "\n"
        print "Distance of route " + str(vehicle_nbr) + ": " + str(route_dist) + " metres"
        print "Demand met by vehicle " + str(vehicle_nbr) + ": " + str(route_demand) + "\n"
    else:
      print 'No solution found.'
  else:
    print 'Specify an instance greater than 0.'
  return vehicle

if __name__ == '__main__':
  #Run main() and output the locations/nodes/waypoints each vehicle serves to
  #'vehicles_route', a dict object. The dict contains {vehicle number/ID: locations
  #served} for each vehicle.
  vehicles_route=main()
  
print "Route Solver  "+ "--- %s seconds ---" % (time.time() - start_time)

##############################Visualisation####################################

#Convert node_index in 'vehicles_route' to the corresponding coordinate and name the new
#dict 'vehicle_coor'.
vehicle_coor={}
for vehicle_id in range(num_vehicles):
    vehicle_coor[vehicle_id]=[locations.tolist()[ii] for ii in vehicles_route[vehicle_id]]
#Then, convert coordinates in 'vehicles_route' of format [Lat,Long] into long,Lat and tie
#the list of coordinates associated with a particular vehicle together, separted by ;, to get ready
#to feed into the URL address for the OSRM API to get the route geometry. That is, 
#vehicle_coor_url={vehicle_id:long1,Lat1;long2,Lat2;..}.
vehicle_coor_url={}
for vehicle_id in range(num_vehicles):
    vehicle_coor_url[vehicle_id]=[]
    dummy1=''
    for vehicle_waypoint in range(len(vehicle_coor[vehicle_id])):
        dummy2=str(vehicle_coor[vehicle_id][vehicle_waypoint][1])+','+\
                str(vehicle_coor[vehicle_id][vehicle_waypoint][0])+';'
        dummy1=dummy1+dummy2
    vehicle_coor_url[vehicle_id]=dummy1[:-1]


#Use OSRM API to get the geometry of a vehicle's route. vehicle_coor_url is a string of node coordinates
#served by the vehicle. Geometry format is GeoJSON; Other formats, e.g. Polyline 6, are possible.
def route_geometry(vehicle_coor_url):
    url_add='http://router.project-osrm.org/route/v1/driving/%s?overview=full&geometries=geojson' %(vehicle_coor_url)
    url_request = Request(url_add)
    url_response = urlopen(url_request)
    url_data = url_response.read()
    url_dict=json.loads(url_data)
    geometry=url_dict['routes'][0]['geometry']['coordinates']
    return geometry 


#Class for generating a thread to get the geometry of the vehicles' route.
#Definitions very similar to those in thread class given above.
class ThreadUrlRoute(threading.Thread):
    def __init__(self, queue,RouteGeometry):
        threading.Thread.__init__(self)
        self.queue = queue

    def run(self):    
        while True:
            vehicleID, vehicleCoorURL = self.queue.get()
            #print(vehicleID)
            RouteGeometry[vehicleID] = route_geometry(vehicleCoorURL)
            self.queue.task_done()
#Add objects to 'queue'
queue = Queue.Queue()         
for vehicle_id in range(num_vehicles):
    queue.put((vehicle_id,vehicle_coor_url[vehicle_id]))
#Initialise the dict 'RouteGeometry', which will collect the route of each vehicle.
#To be passed to the threads.
RouteGeometry={}
for vehicle_id in range(num_vehicles):RouteGeometry[vehicle_id]=''
#Spawn a number of threads, and pass them the 'queue' and 'RouteGeometry', which wil
#be filled with {vehicle_ID:[list of coordinates defining the vehicle route]}.
for i in range(num_vehicles):
    t = ThreadUrlRoute(queue,RouteGeometry)
    t.setDaemon(True)
    t.start()
#Wait on the queue until everything has been processed.
queue.join()

print "Communicate with API (2)  "+ "--- %s seconds ---" % (time.time() - start_time)


#Average location of locations.
AvgLong=np.average(np.array(zip(*locations.tolist())[0]))
AvgLat=np.average(np.array(zip(*locations.tolist())[1]))
#Create (slippy) map tile centred at average location.
map_osm = folium.Map(location=[AvgLong, AvgLat], zoom_start=13)

RouteGeometrySwap={}
for vehicle_id in range(num_vehicles):
    count=0
    #Swap [Long,Lat] format in 'RouteGeometry' to [Lat,Long] format and assign new format of 'RouteGeometry'
    #to 'RouteGeometrySwap'.
    RouteGeometrySwap[vehicle_id]=zip(zip(*RouteGeometry[vehicle_id])[1],zip(*RouteGeometry[vehicle_id])[0])
    #Plot route of 'vehicle_id' in 'map_osm'.
    folium.PolyLine(RouteGeometrySwap[vehicle_id],color=colour[vehicle_id]).add_to(map_osm)
    #Attach markers to nodes/service points associated with 'vehicle_id'.
    for vehicle_waypoint in vehicle_coor[vehicle_id]:
        count+=1
        label_marker='%s' %(count)
        folium.Marker(vehicle_waypoint, icon=folium.Icon(color=colour[vehicle_id],icon='arrow-down'),popup=label_marker).add_to(map_osm)
#Add special marker for depot location
folium.Marker(locations[depot], icon=folium.Icon(color='black',icon='fullscreen'),popup='Depot').add_to(map_osm)

#Save the map in html format. Double click the file to open map in web browser.
map_osm.save('map_osm.html')
print "Create & Save Map  "+ "--- %s seconds ---" % (time.time() - start_time)
print "OSM map saved to '"+ os.getcwd() +"'."