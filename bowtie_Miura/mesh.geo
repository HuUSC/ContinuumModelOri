phi = Pi/6;
L = Sqrt(3) * Cos(phi/2);
H = 2 * Sqrt( 2/ ( 5-3 * Cos(phi) ) );
L = 1;
h = .02; //.02; // mesh size
l = H/10; //to impose bc

Point(1) = {0, 0, 0, h};
Point(4) = {0, H, 0, h};
Point(8) = {L, 0, 0, h};
Point(5) = {L, H, 0, h};
Point(7) = {L, H/2-l, 0, h};
Point(6) = {L, H/2+l, 0, h};
Point(2) = {0, H/2-l, 0, h};
Point(3) = {0, H/2+l, 0, h};


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
