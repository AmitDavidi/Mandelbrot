#define OLC_PGE_APPLICATION
#include "olcPixelGameEngine.h"
#include <math.h>
#include <chrono>
#include <iostream>
#include <cmath>

#include <immintrin.h>

#define SCREEN_SIZE 800

#define ONE_OVER_MAX 1/SCREEN_SIZE

#define ONE_OVER_LOG_2 1.442695041

constexpr int nMaxThreads = 32;

double map_num(double value, double wanted_min, double wanted_max) {
	return ((value * ONE_OVER_MAX) * (wanted_max - wanted_min)) + wanted_min;
}

// double log_2(double x) { return log(x) * ONE_OVER_LOG_2; }

double does_converge(double x_0, double y_0, int iterations) {
	int i = 0; // iter

	double x = 0;
	double y = 0;
	double smoothed;
	double x_2 = 0;
	double y_2 = 0;
	double w = 0;

	while ((i < iterations) && ((x_2 + y_2) < 4.0)) {

		x = x_2 - y_2 + x_0;
		y = w - x_2 - y_2 + y_0;
		x_2 = x * x;
		y_2 = y * y;
		w = (x + y)*(x + y);
		i++;
	}
	if (i == iterations)
		return 0.0;

	else {

		smoothed = log(x_2 + y_2) * ONE_OVER_LOG_2; //log_2(x) = ln(x) / ln(2)
		return sqrt(i + 1 - smoothed);

	}

}



void insert_arr(double *target, double *source) {
	target[0] = source[0];
	target[1] = source[1];
}


class Example : public olc::PixelGameEngine
{
public:

	double COLORS_a[17][3] =
	{
	{66, 30, 15} ,   // 1
	{25, 7, 26}    , // 2
	{9, 1, 47}     , // 3 
	{4, 4, 73}     , // 4
	{0, 7, 100}    , // 5
	{12, 44, 138}  , // 6
	{24, 82, 177}  , // 7
	{57, 125, 209} , // 8
	{134, 181, 229}, // 9
	{211, 236, 248}, // 10
	{241, 233, 191}, // 11
	{248, 201, 95} , // 12
	{255, 170, 0}  , // 13
	{204, 128, 0}  , // 14
	{153, 87, 0}   , // 15
	{106, 52, 3}   // 16
	};
	/*
	double COLORS_b[17][3] =
	{
	{25, 7, 26}    , // 1
	{9, 1, 47}     ,// 2
	{4, 4, 73}     , // 3
	{0, 7, 100}    , // 4
	{12, 44, 138}  , // 5
	{24, 82, 177}  , // 6
	{57, 125, 209} , // 7
	{134, 181, 229}, // 8
	{211, 236, 248}, //9
	{241, 233, 191}, // 10
	{248, 201, 95} , // 11
	{255, 170, 0}  , // 12
	{204, 128, 0}  , // 13
	{153, 87, 0}   , // 14
	{106, 52, 3}   , // 15
	{66, 30, 15}       //16
	};
	*/

	double COLORS_b_minus_a[17][3] = { {-41.0, -23.0, 11.0},
										{-16.0, -6.0, 21.0},
										{-5.0, 3.0, 26.0},
										{-4.0, 3.0, 27.0},
										{12.0, 37.0, 38.0},
										{12.0, 38.0, 39.0},
										{33.0, 43.0, 32.0},
										{77.0, 56.0, 20.0},
										{77.0, 55.0, 19.0},
										{30.0, -3.0, -57.0},
										{7.0, -32.0, -96.0},
										{7.0, -31.0, -95.0},
										{-51.0, -42.0, 0.0},
										{-51.0, -41.0, 0.0},
										{-47.0, -35.0, 3.0},
										{-40.0, -22.0, 12.0} };



	double  x_range[2] = { -2, 2 };
	double  y_range[2] = { -2, 2 };
	double start_draggin_loc_x;
	double start_draggin_loc_y;

	int history_counter = 0;
	double x_history[50][2];
	double y_history[50][2];


	int ITERS = 1024;
	int zoom = 30;
	int  update_flag = 1;
	double x_scale, y_scale;
	int color_num;
	double color_num_double;
	int start1, end1, start2, end2;

	
	double between_fraction, x_mapped, y_mapped, convergence_test;
	double one_minus_between;
	double x_pos1, y_pos1, x_pos2, y_pos2;
	int x_pos;
	int y_pos;
	int rectLayer = 0;
	double ca, cb, cc;
	double* pRes = nullptr;
	

	Example()
	{
		sAppName = "Example";

	}


public:

	void Draw_pix(int x, int y) {
		between_fraction = std::modf(pRes[y*SCREEN_SIZE + x], &color_num_double);
		
		if (color_num_double) {
			
			color_num = color_num_double;
			color_num %= 16;

			
			ca = COLORS_a[color_num][0] + between_fraction * COLORS_b_minus_a[color_num][0];
			cb = COLORS_a[color_num][1] + between_fraction * COLORS_b_minus_a[color_num][1];
			cc = COLORS_a[color_num][2] + between_fraction * COLORS_b_minus_a[color_num][2];
		

			Draw(x, y, olc::Pixel(ca, cb, cc));
		}
		else
			Draw(x, y, olc::Pixel(0,0,0));
	}

	void Create_Fractal_Intrinsic(double y_scale, double x_scale, int x_size, int y_size, int xl, int yl) {
		__m256d _zr, _zi, _cr, _ci, _a, _b, _zr2, _zi2, _two, _four, _mask1, _x_scale, _x_jump, _x_pos_offsets, _x_pos;
		__m256d _n, _iters, _mask2, _c, _one;

		double y_pos = y_range[0];
		int yea = 0;
		_x_scale = _mm256_set1_pd(x_scale);
		_x_jump = _mm256_set1_pd(4.0*x_scale);
		_x_pos_offsets = _mm256_set_pd(0.0, 1.0, 2.0, 3.0);
		
		_x_pos_offsets = _mm256_mul_pd(_x_pos_offsets, _x_scale);
		
		_one = _mm256_set1_pd(1.0);
		_two = _mm256_set1_pd(2.0); // 2.0 2.0 2.0 2.0
		_four = _mm256_set1_pd(4.0); // 2.0 2.0 2.0 2.0

		_iters = _mm256_set1_pd(ITERS);

		double x_left = map_num(xl, x_range[0], x_range[1]);

		for (int y = yl; y < y_size; y++) {
			// Reset x_position
			_a = _mm256_set1_pd(x_left);
			_x_pos = _mm256_add_pd(_a, _x_pos_offsets);

			_ci = _mm256_set1_pd(y_pos);

			for (int x = xl; x < x_size; x += 4)
			{
				_cr = _x_pos;

				// Zreal = 0
				_zr = _mm256_setzero_pd();

				// Zimag = 0
				_zi = _mm256_setzero_pd();

				// nIterations = 0
				_n = _mm256_setzero_pd();


			repeat:
				// Normal: z = (z * z) + c;
				// Manual: a = zr * zr - zi * zi + cr;
				//         b = zr * zi * 2.0 + ci;
				//         zr = a;
				//         zi = b;


				// zr^2 = zr * zr
				_zr2 = _mm256_mul_pd(_zr, _zr);     // zr * zr

				// zi^2 = zi * zi
				_zi2 = _mm256_mul_pd(_zi, _zi);     // zi * zi

				// a = zr^2 - zi^2
				_a = _mm256_sub_pd(_zr2, _zi2);     // a = (zr * zr) - (zi * zi)

				// a = a + cr
				_a = _mm256_add_pd(_a, _cr);        // a = ((zr * zr) - (zi * zi)) + cr



				// b = zr * zi
				_b = _mm256_mul_pd(_zr, _zi);        // b = zr * zi

				// b = b * 2.0 + ci
				// b = b * |2.0|2.0|2.0|2.0| + ci
				_b = _mm256_fmadd_pd(_b, _two, _ci); // b = (zr * zi) * 2.0 + ci

				// zr = a
				_zr = _a;                            // zr = a

				// zi = b
				_zi = _b;                            // zr = b



				// Normal: while (abs(z) < 2.0 && n < iterations)
				// Manual: while ((zr * zr + zi * zi) < 4.0 && n < iterations)


				// a = zr^2 + zi^2
				_a = _mm256_add_pd(_zr2, _zi2);     // a = (zr * zr) + (zi * zi)

				// m1 = if (a < 4.0)
				// m1 = |if(a[3] < 4.0)|if(a[2] < 4.0)|if(a[1] < 4.0)|if(a[0] < 4.0)|
				// m1 = |111111...11111|000000...00000|111111...11111|000000...00000|
				// m1 = |11...11|00...00|11...11|00...00| <- Shortened to reduce typing :P
				_mask1 = _mm256_cmp_pd(_a, _four, _CMP_LT_OQ);

				// m2 = if (iterations > n)
				// m2 = |00...00|11...11|11...11|00...00|
				_mask2 = _mm256_cmp_pd(_iters, _n, _CMP_GT_OQ);

				// m2 = m2 AND m1 = if(a < 4.0 && iterations > n)
				//
				// m2 =    |00...00|11...11|11...11|00...00|
				// m1 = AND|11...11|00...00|11...11|00...00|
				// m2 =    |00...00|00...00|11...11|00...00|
				_mask2 = _mm256_and_pd(_mask2, _mask1);

				//  c = |(int)1|(int)1|(int)1|(int)1| AND m2
				//
				//  c =    |00...01|00...01|00...01|00...01| 
				// m2 = AND|00...00|00...00|11...11|00...00|
				//  c =    |00...00|00...00|00...01|00...00|
				//
				//  c = |(int)0|(int)0|(int)1|(int)0|
				_c = _mm256_and_pd(_one, _mask2);

				// n = n + c
				// n =  |00...24|00...13|00...08|00...21| 
				// c = +|00...00|00...00|00...01|00...00|
				// n =  |00...24|00...13|00...09|00...21| (Increment only applied to 'enabled' element)
				_n = _mm256_add_pd(_n, _c);

				if (_mm256_movemask_pd(_mask2) > 0)
					goto repeat;

				_x_pos = _mm256_add_pd(_x_pos, _x_jump);


				// smoothed
				
				_mask2 = _mm256_cmp_pd(_iters, _n, _CMP_GT_OQ);
				_n = _mm256_and_pd(_n, _mask2);
	
				_n = _mm256_sqrt_pd(_n);
				

				
				// Store the values from _n to a temporary array
				double temp[4];
				_mm256_storeu_pd(temp, _n);

				// Access the individual values from the temporary array
				yea = y * SCREEN_SIZE + x;
				pRes[yea] = temp[3];
				pRes[yea + 1] = temp[2];
				pRes[yea + 2] = temp[1];
				pRes[yea + 3] = temp[0];
				

			


			}
			y_pos += y_scale;
		}


	}

	bool OnUserCreate() override
	{

		//for (int i = 0; i < 17; i++)
		//	printf("{%.1f, %.1f, %.1f},\n", COLORS_b[i][0] - COLORS_a[i][0], COLORS_b[i][1] - COLORS_a[i][1], COLORS_b[i][2] - COLORS_a[i][2]);
		
	

		pRes = (double*)_aligned_malloc(SCREEN_SIZE * SCREEN_SIZE * sizeof(double), 64);

		rectLayer = CreateLayer();
		EnableLayer(rectLayer, true);

		// Called once at the start, so create things here
		return true;
	}
	 



	void Create_Fractal_Threads(double y_scale, double x_scale) {
		int nSectionWidth = SCREEN_SIZE / nMaxThreads; // region for each thread

		std::thread t[nMaxThreads]; //create thread holder

		// void Create_Fractal_Intrinsic(double y_scale, double x_scale, int x_size, int y_size, int x_l, int y_l)
		
		// create threads and activate the fractal calculation function
		for (int i = 0; i < nMaxThreads; i++)
			t[i] = std::thread(&Example::Create_Fractal_Intrinsic, this,
				y_scale, x_scale, nSectionWidth*(i+1), SCREEN_SIZE, nSectionWidth*i, 0);
				

		for (int i = 0; i < nMaxThreads; i++)
			t[i].join();



	}
	
	bool OnUserUpdate(float fElapsedTime) override
	{
		
		x_pos = GetMouseX();
		y_pos = GetMouseY();
		double x11 = map_num(x_pos, x_range[0], x_range[1]);
		double y11 = map_num(y_pos, y_range[0], y_range[1]);


		//Draw rect around mouse
		SetDrawTarget(nullptr);
		Clear(olc::BLANK);
		DrawRect(x_pos -zoom, y_pos -zoom, 2*zoom, 2*zoom, olc::GREY);

		DrawString(0, 120,std::to_string(x11) + ", " + std::to_string(y11), olc::GREY, 2);

		// Handle Iterations control
		if (GetKey(olc::Q).bPressed) {
			ITERS *= 2;
			update_flag = 1;
		}
		if (GetKey(olc::W).bPressed) {
			ITERS *= 0.5;
			update_flag = 1;
		}
		// Handle zoom 

		if (GetKey(olc::A).bHeld) {
			// calculate new fractal space edges based on mouse loc
			start1   = x_pos * 0.1;
			end1     = SCREEN_SIZE * 0.9 + start1; //some zoom math =D
			start2   = y_pos * 0.1;
			end2     = SCREEN_SIZE * 0.9 + start2;
			
			// update new fractal space
			x_range[0] = map_num(start1, x_range[0], x_range[1]);
			x_range[1] = map_num(end1, x_range[0], x_range[1]);
			y_range[0] = map_num(start2, y_range[0], y_range[1]);
			y_range[1] = map_num(end2, y_range[0], y_range[1]);

			update_flag = 1;
		}
		if (GetKey(olc::S).bHeld) {
			// calculate new fractal space edges based on mouse loc
			start1 = -x_pos * 0.1;
			end1 = 1.1*SCREEN_SIZE + start1;
			start2 = -y_pos * 0.1;
			end2 = 1.1*SCREEN_SIZE + start2;

			// update new fractal space
			x_range[0] = map_num(start1, x_range[0], x_range[1]);
			x_range[1] = map_num(end1, x_range[0], x_range[1]);
			y_range[0] = map_num(start2, y_range[0], y_range[1]);
			y_range[1] = map_num(end2, y_range[0], y_range[1]);

			update_flag = 1;
		}

		// handle picture drag
		if (GetMouse(2).bPressed) {
			start_draggin_loc_x = map_num(x_pos, x_range[0], x_range[1]);
			start_draggin_loc_y = map_num(y_pos, y_range[0], y_range[1]);
		}
		if (GetMouse(2).bHeld) {
			// calculate drag offset 
			double x_offset = map_num(x_pos, x_range[0], x_range[1]) - start_draggin_loc_x;
			double y_offset = map_num(y_pos, y_range[0], y_range[1]) - start_draggin_loc_y;
			
			// update the new fractal space accordingly
			x_range[0] -= x_offset;
			x_range[1] -= x_offset;
			y_range[0] -= y_offset;
			y_range[1] -= y_offset;
			update_flag = 1;
		}
	
		if (GetMouse(1).bPressed && history_counter) {
			// get last and update fractal space

			insert_arr(x_range, x_history[history_counter - 1]);
			insert_arr(y_range, y_history[history_counter - 1]);
			history_counter--;
			update_flag = 1;
		}
		else if (GetMouse(1).bPressed) {
			// set fractal space to default
			x_range[0] = -2;
			x_range[1] = 2;
			y_range[0] = -2;
			y_range[1] = 2;

			update_flag = 1; 
		}

 		if (GetMouse(0).bPressed) {
			// remember last
			insert_arr(x_history[history_counter], x_range);
			insert_arr(y_history[history_counter], y_range);
			history_counter++;

			// calculate new 
			x_pos1 = map_num(x_pos - zoom, x_range[0], x_range[1]);
			x_pos2 = map_num(x_pos + zoom, x_range[0], x_range[1]);
			y_pos1 = map_num(y_pos - zoom, y_range[0], y_range[1]);
			y_pos2 = map_num(y_pos + zoom, y_range[0], y_range[1]);
			
			// update new
			x_range[0] = x_pos1;
			x_range[1] = x_pos2;
			y_range[0] = y_pos1;
			y_range[1] = y_pos2;
			
			update_flag = 1;
			
		}

		if (update_flag) {
			auto start = std::chrono::high_resolution_clock::now();

			SetDrawTarget(rectLayer);

			x_scale = (x_range[1] - x_range[0]) * ONE_OVER_MAX; //ONE_OVER_MAX = 1/SCREEN_SIZE
			y_scale = (y_range[1] - y_range[0]) * ONE_OVER_MAX;

			//Create_Fractal_Intrinsic(y_scale, x_scale, SCREEN_SIZE, SCREEN_SIZE, 0, 0);

			Create_Fractal_Threads(y_scale, x_scale); // Use threads
			auto end = std::chrono::high_resolution_clock::now();

			auto start1 = std::chrono::high_resolution_clock::now();
			
			// Draw Picture to screen
			for(int x = 0; x < SCREEN_SIZE; x++)
				for(int y = 0; y < SCREEN_SIZE; y++)
					Draw_pix(x, y);

			auto end1 = std::chrono::high_resolution_clock::now();

			update_flag = 0;


			std::chrono::duration<double> elapsed = end - start;
			std::chrono::duration<double> elapsed1 = end1 - start1;
			DrawString(0, 30, "Time Taken Calc: " + std::to_string(elapsed.count()) + "s", olc::WHITE, 2);
			DrawString(0, 60, "Time Taken Draw: " + std::to_string(elapsed1.count()) + "s", olc::WHITE, 2);
			DrawString(0, 90, "Iterations: " + std::to_string(ITERS), olc::WHITE, 2);
		}
	
		return true;
	}
};


int main()
{
	Example demo;
	if (demo.Construct(SCREEN_SIZE, SCREEN_SIZE, 1, 1))
		demo.Start();

	return 0;
}
