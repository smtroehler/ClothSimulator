#ifdef __APPLE__
#include <ctime>
#endif
#include <iostream>
#include "Cloth.h"
#include "Particle.h"
#include "Spring.h"
#include "MatrixStack.h"
#include "Program.h"
#include "GLSL.h"

using namespace std;

Spring *createSpring(Particle *p0, Particle *p1, double K)
{
	Spring *s = new Spring(p0, p1);
	s->K = K;
	Eigen::Vector3d x0 = p0->x;
	Eigen::Vector3d x1 = p1->x;
	Eigen::Vector3d dx = x1 - x0;
	s->L = dx.norm();
	return s;
}

Cloth::Cloth(int rows, int cols,
			 const Eigen::Vector3d &x00,
			 const Eigen::Vector3d &x01,
			 const Eigen::Vector3d &x10,
			 const Eigen::Vector3d &x11,
			 double mass, double stiffness)
{
	assert(rows > 1);
	assert(cols > 1);
	assert(mass > 0.0);
	assert(stiffness > 0.0);
	
	this->rows = rows;
	this->cols = cols;
	
	// Create particles
	n = 0;
	int nVerts = rows*cols;
	for(int i = 0; i < rows; ++i) {
		double u = i / (rows - 1.0);
		Eigen::Vector3d x0 = (1 - u)*x00 + u*x10;
		Eigen::Vector3d x1 = (1 - u)*x01 + u*x11;
		for(int j = 0; j < cols; ++j) {
			double v = j / (cols - 1.0);
			Eigen::Vector3d x = (1 - v)*x0 + v*x1;
			Particle *p = new Particle();
			particles.push_back(p);
			p->r = 0.01; // Used for collisions
			p->x = x;
			p->v << 0.0, 0.0, 0.0;
			p->m = mass/(nVerts);
			// Pin two particles
			if(i == 0 && (j == 0 || j == cols-1)) {
//			if(i == 0 && j == 0) {
				p->fixed = true;
				p->i = -1;
			} else {
				p->fixed = false;
				p->i = n;
				n += 3;
			}
		}
	}
	
	// Create x springs
	for(int i = 0; i < rows; ++i) {
		for(int j = 0; j < cols-1; ++j) {
			int k0 = i*cols + j;
			int k1 = k0 + 1;
			springs.push_back(createSpring(particles[k0], particles[k1], stiffness));
		}
	}
	
	// Create y springs
	for(int j = 0; j < cols; ++j) {
		for(int i = 0; i < rows-1; ++i) {
			int k0 = i*cols + j;
			int k1 = k0 + cols;
			springs.push_back(createSpring(particles[k0], particles[k1], stiffness));
		}
	}
	
	// Create shear springs
	for(int i = 0; i < rows-1; ++i) {
		for(int j = 0; j < cols-1; ++j) {
			int k00 = i*cols + j;
			int k10 = k00 + 1;
			int k01 = k00 + cols;
			int k11 = k01 + 1;
			springs.push_back(createSpring(particles[k00], particles[k11], stiffness));
			springs.push_back(createSpring(particles[k10], particles[k01], stiffness));
		}
	}
	
	// Build system matrices and vectors
	M.resize(n,n);
	K.resize(n,n);
	v.resize(n);
	f.resize(n);
	
	// Build vertex buffers
	posBuf.clear();
	norBuf.clear();
	texBuf.clear();
	eleBuf.clear();
	posBuf.resize(nVerts*3);
	norBuf.resize(nVerts*3);
	updatePosNor();
	// Texture coordinates (don't change)
	for(int i = 0; i < rows; ++i) {
		for(int j = 0; j < cols; ++j) {
			texBuf.push_back(i/(rows-1.0));
			texBuf.push_back(j/(cols-1.0));
		}
	}
	// Elements (don't change)
	for(int i = 0; i < rows-1; ++i) {
		for(int j = 0; j < cols; ++j) {
			int k0 = i*cols + j;
			int k1 = k0 + cols;
			// Triangle strip
			eleBuf.push_back(k0);
			eleBuf.push_back(k1);
		}
	}
}

Cloth::~Cloth()
{
	while(!springs.empty()) {
		delete springs.back();
		springs.pop_back();
	}
	while(!particles.empty()) {
		delete particles.back();
		particles.pop_back();
	}
}

void Cloth::tare()
{
	for(int k = 0; k < (int)particles.size(); ++k) {
		particles[k]->tare();
	}
}

void Cloth::reset()
{
	for(int k = 0; k < (int)particles.size(); ++k) {
		particles[k]->reset();
	}
	updatePosNor();
}

void Cloth::updatePosNor()
{
	// Position
	for(int i = 0; i < rows; ++i) {
		for(int j = 0; j < cols; ++j) {
			int k = i*cols + j;
			Eigen::Vector3d x = particles[k]->x;
			posBuf[3*k+0] = x(0);
			posBuf[3*k+1] = x(1);
			posBuf[3*k+2] = x(2);
		}
	}
	// Normal
	for(int i = 0; i < rows; ++i) {
		for(int j = 0; j < cols; ++j) {
			// Each particle has four neighbors
			//
			//      v1
			//     /|\
			// u0 /_|_\ u1
			//    \ | /
			//     \|/
			//      v0
			//
			// Use these four triangles to compute the normal
			int k = i*cols + j;
			int ku0 = k - 1;
			int ku1 = k + 1;
			int kv0 = k - cols;
			int kv1 = k + cols;
			Eigen::Vector3d x = particles[k]->x;
			Eigen::Vector3d xu0, xu1, xv0, xv1, dx0, dx1, c;
			Eigen::Vector3d nor(0.0, 0.0, 0.0);
			int count = 0;
			// Top-right triangle
			if(j != cols-1 && i != rows-1) {
				xu1 = particles[ku1]->x;
				xv1 = particles[kv1]->x;
				dx0 = xu1 - x;
				dx1 = xv1 - x;
				c = dx0.cross(dx1);
				nor += c.normalized();
				++count;
			}
			// Top-left triangle
			if(j != 0 && i != rows-1) {
				xu1 = particles[kv1]->x;
				xv1 = particles[ku0]->x;
				dx0 = xu1 - x;
				dx1 = xv1 - x;
				c = dx0.cross(dx1);
				nor += c.normalized();
				++count;
			}
			// Bottom-left triangle
			if(j != 0 && i != 0) {
				xu1 = particles[ku0]->x;
				xv1 = particles[kv0]->x;
				dx0 = xu1 - x;
				dx1 = xv1 - x;
				c = dx0.cross(dx1);
				nor += c.normalized();
				++count;
			}
			// Bottom-right triangle
			if(j != cols-1 && i != 0) {
				xu1 = particles[kv0]->x;
				xv1 = particles[ku1]->x;
				dx0 = xu1 - x;
				dx1 = xv1 - x;
				c = dx0.cross(dx1);
				nor += c.normalized();
				++count;
			}
			nor /= count;
			nor.normalize();
			norBuf[3*k+0] = nor(0);
			norBuf[3*k+1] = nor(1);
			norBuf[3*k+2] = nor(2);
		}
	}
}

#ifdef __APPLE__
clock_t t1, t2;
void tic()
{
	t1 = clock();
}
void toc()
{
	t2 = clock();
	cout << (double)(t2 - t1) / CLOCKS_PER_SEC * 1000 << " ms\n";
}
#endif

void Cloth::step(double h, const Eigen::Vector3d &grav, const std::vector<Particle *> &spheres)
{
	//
	// <COLLISION_DETECTION>
	//
	vector<Spring*> collisions;
	// Create new Spring objects between cloth particles and the collision spheres with "new".
	//    Spring *spring = new Spring(particle, sphere)
	//    collisions.push_back(spring);
	for(int i = 0; i < (int)particles.size(); ++i) {
		for(int j = 0; j < (int)spheres.size(); ++j) {
			// IMPLEMENT ME

         if((particles[i]->x - spheres[j]->x).norm() < 
            particles[i]->r + spheres[j]->r) {
            Spring* in = createSpring(particles[i], spheres[j], 100);
            in->L = particles[i]->r + spheres[j]->r;
            collisions.push_back(in);
         }
		}
	}
	// Make a big vector containing all the springs
	vector<Spring*> springsCollisions;
	// Add cloth springs
	springsCollisions.insert(springsCollisions.end(), springs.begin(), springs.end());
	// Add collision springs
	springsCollisions.insert(springsCollisions.end(), collisions.begin(), collisions.end());
	//
	// </COLLISION_DETECTION>
	//

	//
	// <INTEGRATION>
	//
	
	M.setZero();
	K.setZero();
	v.setZero();
	f.setZero();
	Eigen::Matrix3d I = Eigen::Matrix3d::Identity();

   for(int i = 0; i < particles.size(); i++) {
      int pInd = particles[i]->i;
      if(pInd != -1 ) {
         
         M.block<3,3>(pInd ,pInd ) = particles[i]->m * I; 
         
         v.segment(pInd, 3) = particles[i]->v;
         
         f.segment(pInd, 3) = particles[i]->m * (grav);
         //K.block<3,1>(pInd * 3, )
      }
   }

   for(int i = 0; i < springsCollisions.size(); i++) {
      int pIndA = springsCollisions[i]->p0->i;
      int pIndB = springsCollisions[i]->p1->i;

      // is vector AB
      Eigen::MatrixXd dx = springsCollisions[i]->p1->x - springsCollisions[i]->p0->x;
      double l = (dx).norm();

      double scal = (dx.transpose() * dx)(0,0);
      Eigen::MatrixXd Ks = -springsCollisions[i]->K / (l * l) * 
                  (((l - springsCollisions[i]->L) * scal * I) 
                  + 
                  ((1 - ((l - springsCollisions[i]->L) / l)) * (dx * dx.transpose()) ) );

      Eigen::Vector3d fspring = springsCollisions[i]->K * (l - springsCollisions[i]->L) * dx / l;
      if(pIndA != -1) {
         
         f.segment(pIndA , 3) += fspring;
         
         K.block<3,3>(pIndA , pIndA) += Ks;
      }
      if(pIndB != -1) {
         f.segment(pIndB , 3) += -fspring;
         K.block<3,3>(pIndB , pIndB) += Ks;
      }

      if(pIndA != -1 && pIndB != -1) {
         K.block<3,3>(pIndA , pIndB) -= Ks;
         K.block<3,3>(pIndB , pIndA) -= Ks;
      }
   }

   Eigen::VectorXd solution = (M - h * h * K ).ldlt().solve(M*v + h*f);

   for(int i = 0; i < particles.size(); i++) {
      int pInd = particles[i]->i;
      if(pInd != -1 ) {
         particles[i]->v = solution.segment(pInd , 3);
         particles[i]->x += h * particles[i]->v;
      }
   }
	//
	// </INTEGRATION>
	//
	
	// Update position and normal buffers
	updatePosNor();
	
	// Delete collision springs
	while(!collisions.empty()) {
		delete collisions.back();
		collisions.pop_back();
	}
}

void Cloth::init()
{
	glGenBuffers(1, &posBufID);
	glBindBuffer(GL_ARRAY_BUFFER, posBufID);
	glBufferData(GL_ARRAY_BUFFER, posBuf.size()*sizeof(float), &posBuf[0], GL_DYNAMIC_DRAW);
	
	glGenBuffers(1, &norBufID);
	glBindBuffer(GL_ARRAY_BUFFER, norBufID);
	glBufferData(GL_ARRAY_BUFFER, norBuf.size()*sizeof(float), &norBuf[0], GL_DYNAMIC_DRAW);
	
	glGenBuffers(1, &texBufID);
	glBindBuffer(GL_ARRAY_BUFFER, texBufID);
	glBufferData(GL_ARRAY_BUFFER, texBuf.size()*sizeof(float), &texBuf[0], GL_STATIC_DRAW);
	
	glGenBuffers(1, &eleBufID);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, eleBufID);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, eleBuf.size()*sizeof(unsigned int), &eleBuf[0], GL_STATIC_DRAW);
	
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	
	assert(glGetError() == GL_NO_ERROR);
}

void Cloth::draw(MatrixStack &MV, const Program &p) const
{
	// Draw mesh
	glUniform3fv(p.getUniform("kdFront"), 1, Eigen::Vector3f(1.0, 0.0, 0.0).data());
	glUniform3fv(p.getUniform("kdBack"),  1, Eigen::Vector3f(1.0, 1.0, 0.0).data());
	MV.pushMatrix();
	glUniformMatrix4fv(p.getUniform("MV"), 1, GL_FALSE, MV.topMatrix().data());
	int h_pos = p.getAttribute("vertPos");
	GLSL::enableVertexAttribArray(h_pos);
	glBindBuffer(GL_ARRAY_BUFFER, posBufID);
	glBufferData(GL_ARRAY_BUFFER, posBuf.size()*sizeof(float), &posBuf[0], GL_DYNAMIC_DRAW);
	glVertexAttribPointer(h_pos, 3, GL_FLOAT, GL_FALSE, 0, (const void *)0);
	int h_nor = p.getAttribute("vertNor");
	GLSL::enableVertexAttribArray(h_nor);
	glBindBuffer(GL_ARRAY_BUFFER, norBufID);
	glBufferData(GL_ARRAY_BUFFER, norBuf.size()*sizeof(float), &norBuf[0], GL_DYNAMIC_DRAW);
	glVertexAttribPointer(h_nor, 3, GL_FLOAT, GL_FALSE, 0, (const void *)0);
	int h_tex = p.getAttribute("vertTex");
	GLSL::enableVertexAttribArray(h_tex);
	glBindBuffer(GL_ARRAY_BUFFER, texBufID);
	glVertexAttribPointer(h_tex, 2, GL_FLOAT, GL_FALSE, 0, (const void *)0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, eleBufID);
	for(int i = 0; i < rows; ++i) {
		glDrawElements(GL_TRIANGLE_STRIP, 2*cols, GL_UNSIGNED_INT, (const void *)(2*cols*i*sizeof(unsigned int)));
	}
	GLSL::disableVertexAttribArray(h_tex);
	GLSL::disableVertexAttribArray(h_nor);
	GLSL::disableVertexAttribArray(h_pos);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	MV.popMatrix();
}
