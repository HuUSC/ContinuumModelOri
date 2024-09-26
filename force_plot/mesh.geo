L = 1;
h = .01; // mesh size
l = .1; //to impose bc

Point(1) = {0, 0, 0, h};
Point(4) = {0, L, 0, h};
Point(8) = {L, 0, 0, h};
Point(5) = {L, L, 0, h};
Point(7) = {L, L/2-l, 0, h};
Point(6) = {L, L/2+l, 0, h};
Point(2) = {0, L/2-l, 0, h};
Point(3) = {0, L/2+l, 0, h};


Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,5};
Line(5) = {5,6};
Line(6) = {6,7};
Line(7) = {7,8};
Line(8) = {8,1};
Line Loop(1) = {1:8};

Plane Surface(1) = {1};
Physical Surface(1) = {1};

//For bc
Physical Line(1) = {2};
Physical Line(2) = {6};
