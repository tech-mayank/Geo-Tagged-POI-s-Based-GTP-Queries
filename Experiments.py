import pandas as pd
import numpy as np
import time
import rtree
from rtree import Rtree
from random import randint,sample
import networkx as nx

oldenburg_edges = pd.read_csv('data_oldenburg.txt', sep=" ", header=None)
oldenburg_edges.columns = ["index","node 1","node 2","edge dist"]
oldenburg_nodes = pd.read_csv("data_oldenburg_nodes.txt",sep=" ", header=None)
oldenburg_nodes.columns = ['nodes','x-coord','y-coord']

san_joquin_edges = pd.read_csv('san_joquin_edges.txt', sep=" ", header=None)
san_joquin_edges.columns = ["index","node 1","node 2","edge dist"]
san_joquin_nodes = pd.read_csv("san_joquin_nodes.txt",sep=" ", header=None)
san_joquin_nodes.columns = ['nodes','x-coord','y-coord']

NA_edges = pd.read_csv('NA_edges.txt',sep=" ", header=None)
NA_edges.columns = ["index","node 1","node 2","edge dist"]
NA_nodes = pd.read_csv("NA_nodes.txt",sep=" ", header=None)
NA_nodes.columns = ['nodes','x-coord','y-coord']

main_tags = {1 : "hotel",2 : "beach area",3 : "restaurant",4 : "museum",5 : "party hall",6 : "cinema",7 : "gymnasium",8 : "shopping mall",
             9 : "food corner",10 : "Airport",11 : "railway station",12 : "bus stand",13 : "hospital",14 : "Jewellers",15 : "Parking Area",
             16 : "Municipal Corporation",17 : "Grocers Store",18 : "Electronics",19 : "Fruit Shop",20 : "Industry",21 : "children park",
             22 : "Bank",23 : "Farm",24 : "Vegetble market",25 : "Stationary",26 : "Dairy",27 : "Temple",28 : "Church",29 : "hill station",
             30 : "tourist spot",31 : "Schools",32 : "College"}
list_of_tags = {"hotel" : ["rooms available","terrace area","festive food","Business suite","meeting hall","conference hall","Deluxe room","5-start"],
                "beach area" : ["beds","wine and champaigne","waves","coconut water","oil massage","bikini","beach sand","sunset point"],
                "restaurant" : ["free dinner","kitty parties","chicken special","dining area","family dining","la carte","Buffet","cutlery"],
                "museum" : ["preserved art","mummies","since 1800","antique pieces","explore","war memorial","historical article","war suits"],
                "party hall" : ["late night party","wine","pool party","girls and boys","18+ allowed","ring ceremony","free shots","DJ"],
                "cinema" : ["popcorns","night shows","morning shows","movie combos","corner seats","Audi","large screen","sound system"],
                "gymnasium" : ["proteins","Yoga postures","body builders","balanced diet","workout","Abs and shoulders","Exercise","cycling"],
                "shopping mall" : ["Sale","shopping","discounts","elevators","shops","suits and sarees","Toys","kids corner"],
                "food corner" : ["thai food","chinese","tandoori snacks","mexican","desi chaat","Indian food","South Indian","Masala"],
                "Airport" : ["boarding","luggage area","security check","custom check","immigration","Luggage belt","Boarding pass","Lounge"],
                "railway station" : ["ticket counter","Rajdhani express","luggage area","platforms","railway canteen","Waiting area","Enquiry","Workshop area"],
                "bus stand" : ["Buses","ticket area","Entry","rest room","drivers","Waiting Area","Bus stop","bus terminals"],
                "hospital" : ["doctors","nurses","wounds","medical","oxygen","medicines","beds","Doctors cabin"],
                "Jewellers" : ["Gold chain","24 carat","Siver coins","Diamond necklace","Golden rings","Gold and Diamonds","22 carat","Kundan"],
                "Parking Area" : ["24 hours open","4 wheelers","night parking","day parking","car stand","2 wheelers","cycles and scooters","trucks"],
                "Municipal Corporation" : ["MC","MC mayor","municipality","minicipal workers","govt. officers","waiting area","municipal block","Mayor office"],
                "Grocers Store" : ["pulses","biscuits","rice","grains","pickles","cosmetics","shampoos","soaps"],
                "Electronics" : ["Hair Dryers","Watches","Television","Waching Machines","Microwaves","Ovens","Bulbs and lights","Gadgets"],
                "Fruit Shop" : ["Mangoes","Apples","Red Strawberries","Sweet pineapple","Juicy Fruits","Orang oranges","green grapes","Kiwi"],
                "Industry" : ["workers required","Machines Required","Heavy Machinery","Factory","Worker union","Power house","industry holidays","rules"],
                "children park" : ["swings","pool","Grass","Playing area","children area","Toilets","little toddlers","light swings"],
                "Bank" : ["cash","cash withdrawal","ATM","Transactions","Enquiry","Lockers","Bank Manager","Employyes"],
                "Farm" : ["Farm fresh","Greenery","Fruits","Vegetables","Animals","Fodder","Seeds","Fisheries"],
                "Vegetble market" : ["Leafy veggies","Potatoes","juicy tomatoes","Green vegetables","Onions","frozen peas","Frozen foods","Unhygienic"],
                "Stationary" : ["Pens","pencils","copies","Photostat","Glitters","colours","Papers","pen pencils"],
                "Dairy" : ["Fresh milk","soya milk","desi ghee","butter","cheese","cottage cheese","fresh paneer","yogurt"],
                "Temple" : ["Rings","idols","bells","Priests","Chantings","Mantras","Devotees","silent rooms"],
                "Church" : ["Bells","Father","Bishop","Bible","candles","White church","Christian devotees","jesus"],
                "hill station" : ["Flowers","valleys","Waterfalls","zig zag roads","Hilly animals","Hills","mountains","wild animals"],
                "tourist spot" : ["playing area","Sports","Pools","rooms","Food stalls","resting area","Shops","entertainment"],
                "Schools" : ["students","Playgrounds","classes","Teachers","Staff rooms","Principal","chair tables","Black boards"],
                "College" : ["Students","playing Courts","fields","swimming pools","Dean","administration","hostels","night canteens"]}

tags_oldenburg = {}
for i in range(0,6105):
  tag = randint(1,32)
  temp = []
  temp_tags = sample([0,1,2,3,4,5,6,7], 2)
  for j in temp_tags:
    temp.append(list_of_tags[main_tags[tag]][j])
  tags_oldenburg[i] = temp
  
tags_NA = {}
for i in range(0,175813):
  tag = randint(1,32)
  temp = []
  temp_tags = sample([0,1,2,3,4,5,6,7], 2)
  for j in temp_tags:
    temp.append(list_of_tags[main_tags[tag]][j])
  tags_NA[i] = temp

tags_san_joquin = {}
for i in range(0,18263):
  tag = randint(1,32)
  temp = []
  temp_tags = sample([0,1,2,3,4,5,6,7], 2)
  for j in temp_tags:
    temp.append(list_of_tags[main_tags[tag]][j])
  tags_san_joquin[i] = temp

G_oldenburg = nx.Graph()
G_san_joquin = nx.Graph()
G_NA = nx.Graph()

for i in range(0,6105):
  G_oldenburg.add_node(i)

for i in range(0,18263):
  G_san_joquin.add_node(i)

for i in range(0,175813):
  G_NA.add_node(i)

for i in range(0,7035):
  G_oldenburg.add_edge(oldenburg_edges["node 1"][i],oldenburg_edges["node 2"][i],weight= oldenburg_edges["edge dist"][i])

for i in range(0,23874):
  G_san_joquin.add_edge(san_joquin_edges["node 1"][i],san_joquin_edges["node 2"][i],weight= san_joquin_edges["edge dist"][i])

for i in range(0,179179):
  G_NA.add_edge(NA_edges["node 1"][i],NA_edges["node 2"][i],weight= NA_edges["edge dist"][i])


def compute_dist_matrix_BF(nodes,graph):
  dist_mat = {}
  for i in nodes:
    temp = dict(nx.single_source_dijkstra_path_length(graph,i))
    dist_mat[i] = temp
  return dist_mat

def compute_dist_matrix(source_nodes,nodes,graph):
  # dist_mat = {}
  # for i in nodes:
  #   temp = dict(nx.single_source_dijkstra_path_length(graph,i))
  #   dist_mat[i] = temp
  # return dist_mat
  dist_mat = {}
  for i in nodes:
    temp = {}
    for j in source_nodes:
      temp[j] = nx.dijkstra_path_length(graph,i,j)
    dist_mat[i] = temp
  return dist_mat

def brute_force_2(source,node_tags,destinations,graph,tags):
  start_1 = time.time()
  b = len(source)
  dict_tag_to_node = {}
  temp_list_nodes = []
  for x in node_tags:
    temp = []
    for i in tags:
      if (x in tags[i]):
        temp.append(i)
    dict_tag_to_node[x] = temp
    temp_list_nodes += temp
  dist_mat = compute_dist_matrix_BF(temp_list_nodes,graph)
  min_path_cost = np.inf
  min_path = []
  start = time.time()
  for m in dict_tag_to_node[node_tags[0]]:
    for n in dict_tag_to_node[node_tags[1]]:
        sum1 = 0
        for i in range(0,b):
          sum1 += dist_mat[m][source[i]]
        sum2 = 0
        sum2 += dist_mat[m][n]
        sum3 = 0
        for i in range(0,b):
          sum3 += dist_mat[n][destinations[i]]
        total_cost = sum1 + b*sum2 + sum3
        if (total_cost < min_path_cost):
          min_path_cost = total_cost
          min_path = [m,n]
  end = time.time() 
  return min_path_cost,min_path,end-start,end-start_1

def brute_force_3(source,node_tags,destinations,graph,tags):
  start_1 = time.time()
  b = len(source)
  dict_tag_to_node = {}
  temp_list_nodes = []
  for x in node_tags:
    temp = []
    for i in tags:
      if (x in tags[i]):
        temp.append(i)
    dict_tag_to_node[x] = temp
    temp_list_nodes += temp
  dist_mat = compute_dist_matrix_BF(temp_list_nodes,graph)
  min_path_cost = np.inf
  min_path = []
  start = time.time()
  for m in dict_tag_to_node[node_tags[0]]:
    for n in dict_tag_to_node[node_tags[1]]:
      for p in dict_tag_to_node[node_tags[2]]:
    
        sum1 = 0
        for i in range(0,b):
          sum1 += dist_mat[m][source[i]]
        sum2 = 0
        sum2 += dist_mat[m][n]
        sum2 += dist_mat[n][p]
        sum3 = 0
        for i in range(0,b):
          sum3 += dist_mat[p][destinations[i]]
        total_cost = sum1 + b*sum2 + sum3
        if (total_cost < min_path_cost):
          min_path_cost = total_cost
          min_path = [m,n,p]
  end = time.time()
  return min_path_cost,min_path,end-start,end-start_1

def brute_force_4(source,node_tags,destinations,graph,tags):
  start_1 = time.time()
  b = len(source)
  dict_tag_to_node = {}
  temp_list_nodes = []
  for x in node_tags:
    temp = []
    for i in tags:
      if (x in tags[i]):
        temp.append(i)
    dict_tag_to_node[x] = temp
    temp_list_nodes += temp
  dist_mat = compute_dist_matrix_BF(temp_list_nodes,graph)
  min_path_cost = np.inf
  min_path = []
  start = time.time()
  for m in dict_tag_to_node[node_tags[0]]:
    for n in dict_tag_to_node[node_tags[1]]:
      for p in dict_tag_to_node[node_tags[2]]:
        for q in dict_tag_to_node[node_tags[3]]:
          sum1 = 0
          for i in range(0,b):
            sum1 += dist_mat[m][source[i]]
          sum2 = 0
          sum2 += dist_mat[m][n]
          sum2 += dist_mat[n][p]
          sum2 += dist_mat[p][q]
          sum3 = 0
          for i in range(0,b):
            sum3 += dist_mat[q][destinations[i]]
          total_cost = sum1 + b*sum2 + sum3
          if (total_cost < min_path_cost):
            min_path_cost = total_cost
            min_path = [m,n,p,q]
  end = time.time()
  return min_path_cost,min_path,end-start,end-start_1

graphs = {"oldenburg" : [G_oldenburg,tags_oldenburg],"San Joquin" : [G_san_joquin,tags_san_joquin]}


print("P = 4 and H = 2")
for gr in graphs:
  min_path_cost,min_path,runtime,fruntime = brute_force_2([3,6051,4567,1232],["shopping","dining area"],[78,5342,2345,4563],graphs[gr][0],graphs[gr][1])
  print("for graph :",gr)
  print(min_path_cost,min_path,runtime,fruntime)

print("P = 4 and H = 3")
for gr in graphs:
  min_path_cost,min_path,runtime,fruntime = brute_force_3([3,6051,4567,1232],["shopping","medical","night shows"],[78,5342,2345,4563],graphs[gr][0],graphs[gr][1])
  print("for graph :",gr)
  print(min_path_cost,min_path,runtime,fruntime)

print("P = 4 and H = 4")
for gr in graphs:
  min_path_cost,min_path,runtime,fruntime = brute_force_4([3,6051,4567,1232],["shopping","dining area","night shows","night canteens"],[78,5342,2345,4563],graphs[gr][0],graphs[gr][1])
  print("for graph :",gr)
  print(min_path_cost,min_path,runtime,fruntime)

print("P = 8 and H = 2")
for gr in graphs:
  min_path_cost,min_path,runtime,fruntime = brute_force_2([3,6051,4567,1232,5087,2222,456,3789],["shopping","dining area"],[78,5342,2345,4563,6102,123,1900,3190],graphs[gr][0],graphs[gr][1])
  print("for graph :",gr)
  print(min_path_cost,min_path,runtime,fruntime)

print("P = 8 and H = 3")
for gr in graphs:
  min_path_cost,min_path,runtime,fruntime = brute_force_3([3,6051,4567,1232,5087,2222,456,3789],["shopping","dining area","night shows"],[78,5342,2345,4563,6102,123,1900,3190],graphs[gr][0],graphs[gr][1])
  print("for graph :",gr)
  print(min_path_cost,min_path,runtime,fruntime)

print("P = 8 and H = 4")
for gr in graphs:
  min_path_cost,min_path,runtime,fruntime = brute_force_4([3,6051,4567,1232,5087,2222,456,3789],["shopping","dining area","night shows","night canteens"],[78,5342,2345,4563,6102,123,1900,3190],graphs[gr][0],graphs[gr][1])
  print("for graph :",gr)
  print(min_path_cost,min_path,runtime,fruntime)

print("P = 12 and H = 2")
for gr in graphs:
  min_path_cost,min_path,runtime,fruntime = brute_force_2([3,6051,4567,1232,5087,2222,456,3789,1989,0,2888,4789],["shopping","dining area"],[78,5342,2345,4563,6102,123,1900,3190,3000,1000,4590,6000],graphs[gr][0],graphs[gr][1])
  print("for graph :",gr)
  print(min_path_cost,min_path,runtime,fruntime)

print("P = 12 and H = 3")
for gr in graphs:
  min_path_cost,min_path,runtime,fruntime = brute_force_3([3,6051,4567,1232,5087,2222,456,3789,1989,0,2888,4789],["shopping","dining area","night shows"],[78,5342,2345,4563,6102,123,1900,3190,3000,1000,4590,6000],graphs[gr][0],graphs[gr][1])
  print("for graph :",gr)
  print(min_path_cost,min_path,runtime,fruntime)

print("P = 12 and H = 4")
for gr in graphs:
  min_path_cost,min_path,runtime,fruntime = brute_force_3([3,6051,4567,1232,5087,2222,456,3789,1989,0,2888,4789],["shopping","dining area","night shows","night canteens"],[78,5342,2345,4563,6102,123,1900,3190,3000,1000,4590,6000],graphs[gr][0],graphs[gr][1])
  print("for graph :",gr)
  print(min_path_cost,min_path,runtime,fruntime)


print("calculations for GNN")

def gnn(nodes,hashtag,tags,graph,dist_mat):
  tag_nodes = []
  for i in tags:
    if (hashtag in tags[i]):
      tag_nodes.append(i)
  dist = np.inf
  nearest_node = -1
  for i in tag_nodes:
    temp = 0
    for j in nodes:
      temp += dist_mat[i][j]
    if (temp < dist):
      dist = temp
      nearest_node = i
  return nearest_node

def nn(node,hashtag,tags,graph,dist_mat):
  tag_nodes = []
  for i in tags:
    if (hashtag in tags[i]):
      tag_nodes.append(i)
  dist = np.inf
  nn = -1
  for i in tag_nodes:
    temp = nx.dijkstra_path_length(graph,i,node)
    if (temp < dist):
      dist = temp
      nn = i
  return nn


def gtp_using_gnn_2(source,node_tags,destinations,graph,tags):
  start_1 = time.time()
  temp_list_nodes = []
  temp_list_nodes2 = []
  for x in node_tags:
    temp = []
    for i in tags:
      if (x in tags[i]):
        temp.append(i)
        if (x == node_tags[1]):
          temp_list_nodes2.append(i)
    temp_list_nodes += temp
  source_nodes = source + destinations
  dist_mat = compute_dist_matrix(source_nodes,temp_list_nodes,graph)
  start = time.time()
  p1 = gnn(source,node_tags[0],tags,graph,dist_mat)
  x = destinations
  x.append(p1)
  dist_mat_for_p1 = compute_dist_matrix([p1],temp_list_nodes2,graph)
  for i in temp_list_nodes2:
    dist_mat[i][p1] = dist_mat_for_p1[i][p1]
  p2 = gnn(x,node_tags[1],tags,graph,dist_mat)
  b = len(source)
  sum1 = 0
  for i in range(b):
    sum1 += dist_mat[p1][source[i]]
  sum2 = 0
  sum2 += nx.dijkstra_path_length(graph,p1,p2)
  sum3 = 0
  for i in range(b):
    sum3 += dist_mat[p2][destinations[i]]
  path_cost = sum1 + b*sum2 + sum3
  path = [p1,p2]
  end = time.time()
  return path_cost,path,end-start,end-start_1


def gtp_using_gnn_3(source,node_tags,destinations,graph,tags):
  start_1 = time.time()
  temp_list_nodes = []
  temp_list_nodes2 = []
  for x in node_tags:
    temp = []
    for i in tags:
      if (x in tags[i]):
        temp.append(i)
        if (x == node_tags[2]):
          temp_list_nodes2.append(i)
    temp_list_nodes += temp
  source_nodes = source + destinations
  dist_mat = compute_dist_matrix(source_nodes,temp_list_nodes,graph)
  start = time.time()
  p1 = gnn(source,node_tags[0],tags,graph,dist_mat)
  p2 = nn(p1,node_tags[1],tags,graph,dist_mat)
  x = destinations
  x.append(p2)
  dist_mat_for_p2 = compute_dist_matrix([p2],temp_list_nodes2,graph)
  for i in temp_list_nodes2:
    dist_mat[i][p2] = dist_mat_for_p2[i][p2]
  p3 = gnn(x,node_tags[2],tags,graph,dist_mat)
  b = len(source)
  sum1 = 0
  for i in range(b):
    sum1 += dist_mat[p1][source[i]]
  sum2 = 0
  sum2 += nx.dijkstra_path_length(graph,p1,p2)
  sum2 += nx.dijkstra_path_length(graph,p2,p3)
  sum3 = 0
  for i in range(b):
    sum3 += dist_mat[p3][destinations[i]]
  path_cost = sum1 + b*sum2 + sum3
  path = [p1,p2,p3]
  end = time.time()
  return path_cost,path,end-start,end-start_1

def gtp_using_gnn_4(source,node_tags,destinations,graph,tags):
  start_1 = time.time()
  temp_list_nodes = []
  temp_list_nodes3 = []
  for x in node_tags:
    temp = []
    for i in tags:
      if (x in tags[i]):
        temp.append(i)
        if (x == node_tags[3]):
          temp_list_nodes3.append(i)
    temp_list_nodes += temp
  source_nodes = source + destinations
  dist_mat = compute_dist_matrix(source_nodes,temp_list_nodes,graph)
  start = time.time()
  p1 = gnn(source,node_tags[0],tags,graph,dist_mat)
  p2 = nn(p1,node_tags[1],tags,graph,dist_mat)
  p3 = nn(p2,node_tags[2],tags,graph,dist_mat)
  x = destinations
  x.append(p3)
  dist_mat_for_p3 = compute_dist_matrix([p3],temp_list_nodes3,graph)
  for i in temp_list_nodes3:
    dist_mat[i][p3] = dist_mat_for_p3[i][p3]
  p4 = gnn(x,node_tags[3],tags,graph,dist_mat)
  b = len(source)
  sum1 = 0
  for i in range(b):
    sum1 += dist_mat[p1][source[i]]
  sum2 = 0
  sum2 += nx.dijkstra_path_length(graph,p1,p2)
  sum2 += nx.dijkstra_path_length(graph,p2,p3)
  sum2 += nx.dijkstra_path_length(graph,p3,p4)
  sum3 = 0
  for i in range(b):
    sum3 += dist_mat[p4][destinations[i]]
  path_cost = sum1 + b*sum2 + sum3
  path = [p1,p2,p3,p4]
  end = time.time()
  return path_cost,path,end-start,end-start_1

print("P = 4 and H = 2")
for gr in graphs:
  min_path_cost,min_path,runtime,fruntime = gtp_using_gnn_2([3,6051,4567,1232],["shopping","dining area"],[78,5342,2345,4563],graphs[gr][0],graphs[gr][1])
  print("for graph :",gr)
  print(min_path_cost,min_path,runtime,fruntime)

print("P = 8 and H = 2")
for gr in graphs:
  min_path_cost,min_path,runtime,fruntime = gtp_using_gnn_2([3,6051,4567,1232,5087,2222,456,3789],["shopping","dining area"],[78,5342,2345,4563,6102,123,1900,3190],graphs[gr][0],graphs[gr][1])
  print("for graph :",gr)
  print(min_path_cost,min_path,runtime,fruntime)

print("P = 12 and H = 2")
for gr in graphs:
  min_path_cost,min_path,runtime,fruntime = gtp_using_gnn_2([3,6051,4567,1232,5087,2222,456,3789,1989,0,2888,4789],["shopping","dining area"],[78,5342,2345,4563,6102,123,1900,3190,3000,1000,4590,6000],graphs[gr][0],graphs[gr][1])
  print("for graph :",gr)
  print(min_path_cost,min_path,runtime,fruntime)

print("P = 4 and H = 3")
for gr in graphs:
  min_path_cost,min_path,runtime,fruntime = gtp_using_gnn_3([3,6051,4567,1232],["shopping","dining area","night shows"],[78,5342,2345,4563],graphs[gr][0],graphs[gr][1])
  print("for graph :",gr)
  print(min_path_cost,min_path,runtime,fruntime)

print("P = 8 and H = 3")
for gr in graphs:
  min_path_cost,min_path,runtime,fruntime = gtp_using_gnn_3([3,6051,4567,1232,5087,2222,456,3789],["shopping","dining area","night shows"],[78,5342,2345,4563,6102,123,1900,3190],graphs[gr][0],graphs[gr][1])
  print("for graph :",gr)
  print(min_path_cost,min_path,runtime,fruntime)

print("P = 12 and H = 3")
for gr in graphs:
  min_path_cost,min_path,runtime,fruntime = gtp_using_gnn_3([3,6051,4567,1232,5087,2222,456,3789,1989,0,2888,4789],["shopping","dining area","night shows"],[78,5342,2345,4563,6102,123,1900,3190,3000,1000,4590,6000],graphs[gr][0],graphs[gr][1])
  print("for graph :",gr)
  print(min_path_cost,min_path,runtime,fruntime)

print("P = 4 and H = 4")
for gr in graphs:
  min_path_cost,min_path,runtime,fruntime = gtp_using_gnn_4([3,6051,4567,1232],["shopping","dining area","night shows","night canteens"],[78,5342,2345,4563],graphs[gr][0],graphs[gr][1])
  print("for graph :",gr)
  print(min_path_cost,min_path,runtime,fruntime)

print("P = 8 and H = 4")
for gr in graphs:
  min_path_cost,min_path,runtime,fruntime = gtp_using_gnn_4([3,6051,4567,1232,5087,2222,456,3789],["shopping","dining area","night shows","night canteens"],[78,5342,2345,4563,6102,123,1900,3190],graphs[gr][0],graphs[gr][1])
  print("for graph :",gr)
  print(min_path_cost,min_path,runtime,fruntime)

print("P = 12 and H = 4")
for gr in graphs:
  min_path_cost,min_path,runtime,fruntime = gtp_using_gnn_4([3,6051,4567,1232,5087,2222,456,3789,1989,0,2888,4789],["shopping","dining area","night shows","night canteens"],[78,5342,2345,4563,6102,123,1900,3190,3000,1000,4590,6000],graphs[gr][0],graphs[gr][1])
  print("for graph :",gr)
  print(min_path_cost,min_path,runtime,fruntime)


print("Calculations of rtrees")

def nn_using_rtree(rtree,nodes_list,graph,nodes_data):
  set_of_nn = {-1}
  for i in nodes_list:
    temp_list = list(rtree.nearest((nodes_data['x-coord'][i],nodes_data['y-coord'][i],nodes_data['x-coord'][i],nodes_data['y-coord'][i]), 2))
    # print(temp_list[1:])
    set_of_nn.update(temp_list[1:])
  min_dist = np.inf
  res_node = -1
  set_of_nn.remove(-1)
  dist_mat = compute_dist_matrix(nodes_list,set_of_nn,graph)
  for x in set_of_nn:
    temp_dist = 0
    for i in nodes_list:
      temp_dist += dist_mat[x][i]
    if (temp_dist < min_dist):
      min_dist = temp_dist
      res_node = x
  return res_node

def gtp_using_rtree_2(source,node_tags,destinations,graph,nodes_data,graph_tags):
  start = time.time()
  b = len(source)
  dict_tag_to_node = {}
  for x in node_tags:
    temp = []
    for i in graph_tags:
      if (x in graph_tags[i]):
        temp.append(i)
    dict_tag_to_node[x] = temp
  rtrees = {}
  for x in node_tags:
    temp = Rtree()
    for i in dict_tag_to_node[x]:
      temp.add(i,(nodes_data['x-coord'][i],nodes_data['y-coord'][i],nodes_data['x-coord'][i],nodes_data['y-coord'][i]))
    rtrees[x] = temp
  start = time.time()
  l1 = nn_using_rtree(rtrees[node_tags[0]],source,graph,nodes_data)
  # l2 = list(rtrees[node_tags[1]].nearest((nodes_data['x-coord'][l1],nodes_data['y-coord'][l1],nodes_data['x-coord'][l1],nodes_data['y-coord'][l1]), 2))[1]
  temp_dest = destinations
  temp_dest.append(l1)
  l2 = nn_using_rtree(rtrees[node_tags[1]],temp_dest,graph,nodes_data)
  sum1 = 0
  for i in range(b):
    sum1 += nx.dijkstra_path_length(graph,source[i],l1)
  sum2 = 0
  sum2 += nx.dijkstra_path_length(graph,l1,l2)
  # sum2 += nx.dijkstra_path_length(graph,l2,l3)
  sum3 = 0
  for i in range(b):
    sum3 += nx.dijkstra_path_length(graph,l2,destinations[i])
  
  path_cost = sum1 + b*sum2 + sum3
  path = [l1,l2]
  end = time.time()
  return path_cost,path,end-start


def gtp_using_rtree_3(source,node_tags,destinations,graph,nodes_data,graph_tags):
  start = time.time()
  b = len(source)
  dict_tag_to_node = {}
  for x in node_tags:
    temp = []
    for i in graph_tags:
      if (x in graph_tags[i]):
        temp.append(i)
    dict_tag_to_node[x] = temp
  rtrees = {}
  for x in node_tags:
    temp = Rtree()
    for i in dict_tag_to_node[x]:
      temp.add(i,(nodes_data['x-coord'][i],nodes_data['y-coord'][i],nodes_data['x-coord'][i],nodes_data['y-coord'][i]))
    rtrees[x] = temp
  start = time.time()
  l1 = nn_using_rtree(rtrees[node_tags[0]],source,graph,nodes_data)
  l2 = list(rtrees[node_tags[1]].nearest((nodes_data['x-coord'][l1],nodes_data['y-coord'][l1],nodes_data['x-coord'][l1],nodes_data['y-coord'][l1]), 2))[1]
  temp_dest = destinations
  temp_dest.append(l2)
  l3 = nn_using_rtree(rtrees[node_tags[2]],temp_dest,graph,nodes_data)
  sum1 = 0
  for i in range(b):
    sum1 += nx.dijkstra_path_length(graph,source[i],l1)
  sum2 = 0
  sum2 += nx.dijkstra_path_length(graph,l1,l2)
  sum2 += nx.dijkstra_path_length(graph,l2,l3)
  sum3 = 0
  for i in range(b):
    sum3 += nx.dijkstra_path_length(graph,l3,destinations[i])
  
  path_cost = sum1 + b*sum2 + sum3
  path = [l1,l2,l3]
  end = time.time()
  return path_cost,path,end-start


def gtp_using_rtree_4(source,node_tags,destinations,graph,nodes_data,graph_tags):
  start = time.time()
  b = len(source)
  dict_tag_to_node = {}
  for x in node_tags:
    temp = []
    for i in graph_tags:
      if (x in graph_tags[i]):
        temp.append(i)
    dict_tag_to_node[x] = temp
  rtrees = {}
  for x in node_tags:
    temp = Rtree()
    for i in dict_tag_to_node[x]:
      temp.add(i,(nodes_data['x-coord'][i],nodes_data['y-coord'][i],nodes_data['x-coord'][i],nodes_data['y-coord'][i]))
    rtrees[x] = temp
  start = time.time()
  l1 = nn_using_rtree(rtrees[node_tags[0]],source,graph,nodes_data)
  l2 = list(rtrees[node_tags[1]].nearest((nodes_data['x-coord'][l1],nodes_data['y-coord'][l1],nodes_data['x-coord'][l1],nodes_data['y-coord'][l1]), 2))[1]
  l3 = list(rtrees[node_tags[2]].nearest((nodes_data['x-coord'][l2],nodes_data['y-coord'][l2],nodes_data['x-coord'][l2],nodes_data['y-coord'][l2]), 2))[1]
  temp_dest = destinations
  temp_dest.append(l3)
  l4 = nn_using_rtree(rtrees[node_tags[3]],temp_dest,graph,nodes_data)
  sum1 = 0
  for i in range(b):
    sum1 += nx.dijkstra_path_length(graph,source[i],l1)
  sum2 = 0
  sum2 += nx.dijkstra_path_length(graph,l1,l2)
  sum2 += nx.dijkstra_path_length(graph,l2,l3)
  sum2 += nx.dijkstra_path_length(graph,l3,l4)
  sum3 = 0
  for i in range(b):
    sum3 += nx.dijkstra_path_length(graph,l4,destinations[i])
  
  path_cost = sum1 + b*sum2 + sum3
  path = [l1,l2,l3,l4]
  end = time.time()
  return path_cost,path,end-start

graphs = {"oldenburg" : [G_oldenburg,tags_oldenburg,oldenburg_nodes],"San Joquin" : [G_san_joquin,tags_san_joquin,san_joquin_nodes],"NA" : [G_NA,tags_NA,NA_nodes]}

print("P = 4 and H = 2")
for gr in graphs:
  min_path_cost,min_path,runtime = gtp_using_rtree_2([3,6051,4567,1232],["shopping","dining area"],[78,5342,2345,4563],graphs[gr][0],graphs[gr][2],graphs[gr][1])
  print("for graph :",gr)
  print(min_path_cost,min_path,runtime)

print("P = 4 and H = 3")
for gr in graphs:
  min_path_cost,min_path,runtime = gtp_using_rtree_3([3,6051,4567,1232],["shopping","dining area","night shows"],[78,5342,2345,4563],graphs[gr][0],graphs[gr][2],graphs[gr][1])
  print("for graph :",gr)
  print(min_path_cost,min_path,runtime)

print("P = 4 and H = 4")
for gr in graphs:
  min_path_cost,min_path,runtime = gtp_using_rtree_4([3,6051,4567,1232],["shopping","dining area","night shows","night canteens"],[78,5342,2345,4563],graphs[gr][0],graphs[gr][2],graphs[gr][1])
  print("for graph :",gr)
  print(min_path_cost,min_path,runtime)

print("P = 8 and H = 2")
for gr in graphs:
  min_path_cost,min_path,runtime = gtp_using_rtree_2([3,6051,4567,1232,5087,2222,456,3789],["shopping","dining area"],[78,5342,2345,4563,6102,123,1900,3190],graphs[gr][0],graphs[gr][2],graphs[gr][1])
  print("for graph :",gr)
  print(min_path_cost,min_path,runtime)

print("P = 8 and H = 3")
for gr in graphs:
  min_path_cost,min_path,runtime = gtp_using_rtree_3([3,6051,4567,1232,5087,2222,456,3789],["shopping","dining area","night shows"],[78,5342,2345,4563,6102,123,1900,3190],graphs[gr][0],graphs[gr][2],graphs[gr][1])
  print("for graph :",gr)
  print(min_path_cost,min_path,runtime)

print("P = 8 and H = 4")
for gr in graphs:
  min_path_cost,min_path,runtime = gtp_using_rtree_4([3,6051,4567,1232,5087,2222,456,3789],["shopping","dining area","night shows","night canteens"],[78,5342,2345,4563,6102,123,1900,3190],graphs[gr][0],graphs[gr][2],graphs[gr][1])
  print("for graph :",gr)
  print(min_path_cost,min_path,runtime)

print("P = 12 and H = 2")
for gr in graphs:
  min_path_cost,min_path,runtime = gtp_using_rtree_2([3,6051,4567,1232,5087,2222,456,3789,1989,0,2888,4789],["shopping","dining area"],[78,5342,2345,4563,6102,123,1900,3190,3000,1000,4590,6000],graphs[gr][0],graphs[gr][2],graphs[gr][1])
  print("for graph :",gr)
  print(min_path_cost,min_path,runtime)

print("P = 12 and H = 3")
for gr in graphs:
  min_path_cost,min_path,runtime = gtp_using_rtree_3([3,6051,4567,1232,5087,2222,456,3789,1989,0,2888,4789],["shopping","dining area","night shows"],[78,5342,2345,4563,6102,123,1900,3190,3000,1000,4590,6000],graphs[gr][0],graphs[gr][2],graphs[gr][1])
  print("for graph :",gr)
  print(min_path_cost,min_path,runtime)

print("P = 12 and H = 4")
for gr in graphs:
  min_path_cost,min_path,runtime = gtp_using_rtree_4([3,6051,4567,1232,5087,2222,456,3789,1989,0,2888,4789],["shopping","dining area","night shows","night canteens"],[78,5342,2345,4563,6102,123,1900,3190,3000,1000,4590,6000],graphs[gr][0],graphs[gr][2],graphs[gr][1])
  print("for graph :",gr)
  print(min_path_cost,min_path,runtime)

def random(graph,hashtag,tags):
  nodes_of_hashtag = []
  for i in tags:
    if (hashtag in tags[i]):
      nodes_of_hashtag.append(i)
  temp = randint(0,len(nodes_of_hashtag)-1)
  return nodes_of_hashtag[temp]
    
def heighest_degree(graph,hashtag,tags):
  nodes_of_hashtag = []
  for i in tags:
    if (hashtag in tags[i]):
      nodes_of_hashtag.append(i)
  degree = 0
  ans = -1
  for i in nodes_of_hashtag:
    if (graph.degree(i) > degree):
      degree = graph.degree(i)
      ans = i
  return ans

def random(graph,hashtag,tags):
  nodes_of_hashtag = []
  for i in tags:
    if (hashtag in tags[i]):
      nodes_of_hashtag.append(i)
  temp = randint(0,len(nodes_of_hashtag)-1)
  return nodes_of_hashtag[temp]
    
def heighest_degree(graph,hashtag,tags):
  nodes_of_hashtag = []
  for i in tags:
    if (hashtag in tags[i]):
      nodes_of_hashtag.append(i)
  degree = 0
  ans = -1
  for i in nodes_of_hashtag:
    if (graph.degree(i) > degree):
      degree = graph.degree(i)
      ans = i
  return ans

def balanced_degree(graph,hashtag1,hashtag2,hashtag3,tags):
  nodes_tags_tag1 = []
  nodes_tags_tag2 = []
  nodes_tags_tag3 = []
  for x in tags:
    if (hashtag1 in tags[x]):
      nodes_tags_tag1.append(x)
    if (hashtag2 in tags[x]):
      nodes_tags_tag2.append(x)
    if (hashtag3 in tags[x]):
      nodes_tags_tag3.append(x)
  BD_nodes = {}
  for x in nodes_tags_tag2:
    BD_nodes[x] = 0
  for x in nodes_tags_tag2:
    for y in nodes_tags_tag1:
      if (graph.has_edge(x,y)):
        BD_nodes[x] = BD_nodes[x] + 1
    for z in nodes_tags_tag3:
      if (graph.has_edge(x,z)):
        BD_nodes[x] = BD_nodes[x] + 1
  max_node = -1
  res = -1
  for k in BD_nodes:
    if (BD_nodes[k] > max_node):
      max_node = BD_nodes[k]
      res = k
  return k

# hashtags_query = [["shopping","dining area"],["shopping","dining area","night shows"],["shopping","dining area","night shows","night canteens"]]
graphs = {"oldenburg" : [G_oldenburg,tags_oldenburg,oldenburg_nodes],"San Joquin" : [G_san_joquin,tags_san_joquin,san_joquin_nodes],"NA" : [G_NA,tags_NA,NA_nodes]}

hashtags_query = [
[[3,6051,4567,1232],["shopping","dining area"],[78,5342,2345,4563]],
[[3,6051,4567,1232],["shopping","dining area","night shows"],[78,5342,2345,4563]],
[[3,6051,4567,1232],["shopping","dining area","night shows","night canteens"],[78,5342,2345,4563]],
[[3,6051,4567,1232,5087,2222,456,3789],["shopping","dining area"],[78,5342,2345,4563,6102,123,1900,3190]],
[[3,6051,4567,1232,5087,2222,456,3789],["shopping","dining area","night shows"],[78,5342,2345,4563,6102,123,1900,3190]],
[[3,6051,4567,1232,5087,2222,456,3789],["shopping","dining area","night shows","night canteens"],[78,5342,2345,4563,6102,123,1900,3190]],
[[3,6051,4567,1232,5087,2222,456,3789,1989,0,2888,4789],["shopping","dining area"],[78,5342,2345,4563,6102,123,1900,3190,3000,1000,4590,6000]],
[[3,6051,4567,1232,5087,2222,456,3789,1989,0,2888,4789],["shopping","dining area","night shows"],[78,5342,2345,4563,6102,123,1900,3190,3000,1000,4590,6000]],
[[3,6051,4567,1232,5087,2222,456,3789,1989,0,2888,4789],["shopping","dining area","night shows","night canteens"],[78,5342,2345,4563,6102,123,1900,3190,3000,1000,4590,6000]]]

for gr in graphs:
  for i in hashtags_query:
    temp = []
    start = time.time()
    for j in range(len(i[1])):
      x = random(graphs[gr][0],i[1][j],graphs[gr][1])
      temp.append(x)
    sum1 = 0
    for x in i[0]:
        sum1 += nx.dijkstra_path_length(graphs[gr][0],x,temp[0])
    sum2 = 0
    for j in range(len(temp)-1):
        sum2 += nx.dijkstra_path_length(graphs[gr][0],temp[j],temp[j+1])
    sum3 = 0
    for x in i[2]:
        sum3 += nx.dijkstra_path_length(graphs[gr][0],x,temp[len(temp)-1])
    total_dist = sum1 + len(i[0])*sum2 + sum3
    end = time.time()
    print(gr,len(i[0]),len(i[1]),total_dist,temp,end-start)

for gr in graphs:
  for i in hashtags_query:
    temp = []
    start = time.time()
    for j in range(len(i[1])):
      x = heighest_degree(graphs[gr][0],i[1][j],graphs[gr][1])
      temp.append(x)
    sum1 = 0
    for x in i[0]:
        sum1 += nx.dijkstra_path_length(graphs[gr][0],x,temp[0])
    sum2 = 0
    for j in range(len(temp)-1):
        sum2 += nx.dijkstra_path_length(graphs[gr][0],temp[j],temp[j+1])
    sum3 = 0
    for x in i[2]:
        sum3 += nx.dijkstra_path_length(graphs[gr][0],x,temp[len(temp)-1])
    total_dist = sum1 + len(i[0])*sum2 + sum3
    end = time.time()
    print(gr,len(i[0]),len(i[1]),total_dist,end-start)

for gr in graphs:
  for i in hashtags_query:
    temp = []
    start = time.time()
    x = heighest_degree(graphs[gr][0],i[1][0],graphs[gr][1])
    temp.append(x)
    for j in range(1,len(i[1])-1):
      x = balanced_degree(graphs[gr][0],i[1][j-1],i[1][j],i[1][j+1],graphs[gr][1])
      temp.append(x)
    x = heighest_degree(graphs[gr][0],i[1][len(i[1])-1],graphs[gr][1])
    temp.append(x)
    sum1 = 0
    for x in i[0]:
        sum1 += nx.dijkstra_path_length(graphs[gr][0],x,temp[0])
    sum2 = 0
    for j in range(len(temp)-1):
        sum2 += nx.dijkstra_path_length(graphs[gr][0],temp[j],temp[j+1])
    sum3 = 0
    for x in i[2]:
        sum3 += nx.dijkstra_path_length(graphs[gr][0],x,temp[len(temp)-1])
    total_dist = sum1 + len(i[0])*sum2 + sum3
    end = time.time()
    print(gr,len(i[0]),len(i[1]),total_dist,end-start)



