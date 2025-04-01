#ifndef FILE_IFGFOPERATOR
#define FILE_IFGFOPERATOR

#include <basematrix.hpp>
#include <solve.hpp>

#include <Eigen/Dense>

#include <cstdlib>
#include <tbb/task_arena.h>
#include <tbb/global_control.h>
#include <fenv.h>
#include <fstream>

#include "fmmoperator.hpp"
#include "ngbem.hpp"

#include "ifgf_library.hpp"

#include "nearfieldoperator.hpp"

typedef std::complex<double> Complex;

namespace ngbem
{

    template<typename KERNEL>
    class  IFGF_Operator : public FMM_Operator<KERNEL >
    {
    public:
	IFGF_Operator(KERNEL _kernel, Array<Vec<3> > _xpts, Array<Vec<3> > _ypts,
		      Array<Vec<3>> _xnv, Array<Vec<3>> _ynv,const BEMParameters& param, shared_ptr<BaseMatrix> _nfop,
		      shared_ptr<BaseMatrix> _evalx,
		      shared_ptr<BaseMatrix> _evaly)
	    :
	    FMM_Operator<KERNEL>(_kernel,std::move(_xpts), std::move( _ypts), std::move(_xnv), std::move(_ynv))
	{

	}
	
	void SetNearfield(shared_ptr<BaseMatrix> _nfop)
      {
	
      }

      
    };

    
#ifdef USE_IFGF

//#include <grad_helmholtz_ifgf.hpp>
//#include <combined_field_helmholtz_ifgf.hpp>
//#include <laplace_ifgf.hpp>



    
  template<>
  class IFGF_Operator<HelmholtzSLKernel<3> > : public Base_FMM_Operator<std::complex<double> > 
  {
      typedef HelmholtzSLKernel<3>  KERNEL;
      typedef HelmholtzIfgfOperator3d OperatorType;
      typedef Base_FMM_Operator<std::complex<double > > BASE;
      using value_type = typename  KERNEL::value_type;  
      
      bool do_nearfield;
  protected:
      std::unique_ptr<OperatorType> op;
      KERNEL kernel;
      shared_ptr<BaseMatrix> nfop;
      
      shared_ptr<BaseMatrix> evalx;
      shared_ptr<BaseMatrix> evaly;


  public:
      IFGF_Operator(KERNEL _kernel, Array<Vec<3> > _xpts, Array<Vec<3> > _ypts,
		    Array<Vec<3>> _xnv, Array<Vec<3>> _ynv, const BEMParameters& param, shared_ptr<BaseMatrix> _nfop,
		    shared_ptr<BaseMatrix> _evalx,
		    shared_ptr<BaseMatrix> _evaly
		    )
	  : BASE(std::move(_xpts), std::move( _ypts), std::move(_xnv), std::move(_ynv)),
	    kernel(_kernel),
	    do_nearfield(true),
	    nfop(_nfop),
	    evalx(_evalx),
	    evaly(_evaly)
      
      {
	  std::cout<<"creating ifgf op"<<std::endl;


	  size_t leafSize=param.leafsize;
	  size_t order=param.expansion_order;
	  int n_elem=param.n_elements;
	  double tol=param.eps;
	  double waveNumber=_kernel.GetKappa();


	  op=make_unique<HelmholtzIfgfOperator3d > (waveNumber,leafSize,order,n_elem,tol);
	  

	  op->init(xpts[0].Data(), xpts.Size(),ypts[0].Data(),ypts.Size());	
      }


      void  Mult(const BaseVector & x, BaseVector & y) const 
      {
	  std::cout<<"ifgf mult"<<std::endl;
	  static Timer tall("ngbem fmm apply HelmholtzCF (IFGF)"); RegionTimer reg(tall);

	  auto tmp = VVector<value_type>(ypts.Size());
	  auto ftmp = tmp.template FV<value_type >();

	  auto tmpx = VVector<value_type>(xpts.Size());

	  tmpx=0;
	  evalx->Mult(x,tmpx);
	  
	  auto fx = tmpx.FV<Complex>();

	  y=0;
	  tmp=0;
	  //auto global_control = tbb::global_control( tbb::global_control::max_allowed_parallelism,      12);                                                                                                                                                                                   

	  //all of the work is done one the GPU anyway, we just need the CPU to supervise. 
	  std::thread t1([&]() {
	      op->mult(fx.Data(),fx.Size(),ftmp.Data(),ftmp.Size());
	  });

	  nfop->Mult(x,y);


	  t1.join();	  
	  
	  //y+=TransposeOperator(evaly)*tmp;
	  evaly->MultTransAdd(1,tmp,y);
	  
      }


      void SetNearfield(shared_ptr<BaseMatrix> _nfop)
      {
	  nfop=_nfop;
      }


  };

      template<>
  class IFGF_Operator<ModifiedHelmholtzSLKernel<3> > : public Base_FMM_Operator<std::complex<double> > 
  {
      typedef ModifiedHelmholtzSLKernel<3>  KERNEL;
      typedef ModifiedHelmholtzIfgfOperator3d OperatorType;
      typedef Base_FMM_Operator<std::complex<double > > BASE;

      using value_type = typename  KERNEL::value_type;  

  protected:
      std::unique_ptr<OperatorType> op;
      KERNEL kernel;
      shared_ptr<BaseMatrix> nfop;

      shared_ptr<BaseMatrix> evalx;
      shared_ptr<BaseMatrix> evaly;


  public:
      IFGF_Operator(KERNEL _kernel, Array<Vec<3> > _xpts, Array<Vec<3> > _ypts,
		    Array<Vec<3>> _xnv, Array<Vec<3>> _ynv, const BEMParameters& param, shared_ptr<BaseMatrix> _nfop,
		    shared_ptr<BaseMatrix> _evalx,
		    shared_ptr<BaseMatrix> _evaly
		    )
	  : BASE(std::move(_xpts), std::move( _ypts), std::move(_xnv), std::move(_ynv)),
	  kernel(_kernel),
	  nfop(_nfop),
	  evalx(_evalx),
	  evaly(_evaly)
      {
	  std::cout<<"creating ifgf opitty"<<std::endl;


	  size_t leafSize=param.leafsize;
	  size_t order=param.expansion_order;
	  int n_elem=param.n_elements;
	  double tol=param.eps;
	  Complex waveNumber=_kernel.GetKappa();
	  double maxk=param.maxk;

	  std::cout<<"size="<<xpts.Size()<<std::endl;
	  std::cout<<"size="<<ypts.Size()<<std::endl;


	  //auto global_control = tbb::global_control( tbb::global_control::max_allowed_parallelism,      12);

	  
	  op=make_unique<ModifiedHelmholtzIfgfOperator3d > (std::complex<RealScalar>(waveNumber),leafSize,order,n_elem,tol,maxk);
	  

	  op->init(xpts[0].Data(), xpts.Size(),ypts[0].Data(),ypts.Size());	

	  /*op=make_unique<GradHelmholtzIfgfOperator<3> > (waveNumber,leafSize,order,n_elem,tol);
	  op->setDx(-1);
	  
	  auto srcs=Eigen::Map<typename OperatorType::PointArray>( xpts[0].Data(),3, xpts.Size());
	  auto targets=Eigen::Map<typename OperatorType::PointArray>(ypts[0].Data(),3, ypts.Size());
	  op->init(srcs,targets); */


      }


            void  Mult(const BaseVector & x, BaseVector & y) const 
      {
	  std::cout<<"ifgf mult"<<std::endl;
	  static Timer tall("ngbem fmm apply HelmholtzCF (IFGF)"); RegionTimer reg(tall);

	  auto tmp = VVector<value_type>(ypts.Size());
	  auto ftmp = tmp.template FV<value_type >();

	  auto tmpx = VVector<value_type>(xpts.Size());

	  tmpx=0;
	  evalx->Mult(x,tmpx);
	  
	  auto fx = tmpx.FV<Complex>();

	  y=0;
	  tmp=0;
	  //auto global_control = tbb::global_control( tbb::global_control::max_allowed_parallelism,      12);                                                                                                                                                                                   

	  //all of the work is done one the GPU anyway, we just need the CPU to supervise. 
	  std::thread t1([&]() {
	      op->mult(fx.Data(),fx.Size(),ftmp.Data(),ftmp.Size());
	  });

	  nfop->Mult(x,y);


	  t1.join();	  
	  
	  //y+=TransposeOperator(evaly)*tmp;
	  evaly->MultTransAdd(1,tmp,y);
	  
      }


      void SetNearfield(shared_ptr<BaseMatrix> _nfop)
      {
	  nfop=_nfop;
      }



  };

#if 0


    template<>
  class IFGF_Operator<CombinedFieldKernel<3> > : public Base_FMM_Operator<std::complex<double> > 
  {
      typedef CombinedFieldKernel<3>  KERNEL;
      typedef CombinedFieldHelmholtzIfgfOperator<3> OperatorType;
      typedef Base_FMM_Operator<std::complex<double > > BASE;

  protected:
      std::unique_ptr<OperatorType> op;
      KERNEL kernel;

  public:
      IFGF_Operator(KERNEL _kernel, Array<Vec<3> > _xpts, Array<Vec<3> > _ypts,
		    Array<Vec<3>> _xnv, Array<Vec<3>> _ynv,const BEMParameters& param)
	  : BASE(std::move(_xpts), std::move( _ypts), std::move(_xnv), std::move(_ynv)),
	    kernel(_kernel)
      {
	  std::cout<<"creating ifgf cf op"<<std::endl;
	  double waveNumber=_kernel.GetKappa();
	  size_t leafSize=param.leafsize;
	  size_t order=param.expansion_order;
	  int n_elem=2;
	  double tol=param.eps;
	  
	  std::cout<<"size="<<xpts.Size()<<std::endl;
	  std::cout<<"size="<<ypts.Size()<<std::endl;


	  op=make_unique<CombinedFieldHelmholtzIfgfOperator<3> > (waveNumber,leafSize,order,n_elem,tol);
	  
	  auto srcs=Eigen::Map<typename OperatorType::PointArray>( xpts[0].Data(),3, xpts.Size());
	  auto targets=Eigen::Map<typename OperatorType::PointArray>(ypts[0].Data(),3, ypts.Size());
	  
	  auto src_normals=Eigen::Map<typename OperatorType::PointArray>(xnv[0].Data(),3, xnv.Size());
	  
	  op->init(srcs,targets,src_normals);
      }


      void  Mult(const BaseVector & x, BaseVector & y) const 
      {
	  std::cout<<"ifgf mult"<<std::endl;
	  static Timer tall("ngbem fmm apply CombinedField (IFGF)"); RegionTimer reg(tall);
	  auto fx = x.FV<Complex>();
	  auto fy = y.FV<Complex>();

	  //fy = 0;


	  auto weights=Eigen::Map< Eigen::Vector<std::complex<double>, Eigen::Dynamic> >(fx.Data(),fx.Size());
	  auto results=op->mult(weights);


	  auto y_map=Eigen::Map< Eigen::Vector<std::complex<double>, Eigen::Dynamic> >(fy.Data(),fy.Size());
	  y_map=results;
      }

  };




    template<>
  class IFGF_Operator<LaplaceSLKernel<3> > : public Base_FMM_Operator<double > 
  {
      typedef LaplaceSLKernel<3>  KERNEL;
      typedef LaplaceIfgfOperator<3> OperatorType;
      typedef Base_FMM_Operator<double > BASE;

  protected:
      std::unique_ptr<OperatorType> op;
      KERNEL kernel;
      //Array<Vec<3>> xpts, ypts, xnv, ynv;
  public:
      IFGF_Operator(KERNEL _kernel, Array<Vec<3> > _xpts, Array<Vec<3> > _ypts,
		    Array<Vec<3>> _xnv, Array<Vec<3>> _ynv,const BEMParameters& param)
      : BASE(std::move(_xpts), std::move( _ypts), std::move(_xnv), std::move(_ynv)),
	kernel(_kernel)
    {
	std::cout<<"creating ifgf op"<<std::endl;

	size_t leafSize=param.leafsize;
	size_t order=param.expansion_order;
	int n_elem=1;
	double tol=param.eps;

	std::cout<<"size="<<xpts.Size()<<std::endl;
	std::cout<<"size="<<ypts.Size()<<std::endl;


	op=make_unique<LaplaceIfgfOperator<3> > (leafSize,order,n_elem,tol);

	auto srcs=Eigen::Map<typename OperatorType::PointArray>( xpts[0].Data(),3, xpts.Size());
	auto targets=Eigen::Map<typename OperatorType::PointArray>(ypts[0].Data(),3, ypts.Size());
	    
	op->init(srcs,targets);
	
    }


  void  Mult(const BaseVector & x, BaseVector & y) const 
  {
    std::cout<<"ifgf mult"<<std::endl;
    static Timer tall("ngbem fmm apply LaplaceSL (IFGF)"); RegionTimer reg(tall);
    auto fx = x.FV<double>();
    auto fy = y.FV<double>();

    //fy = 0;


    auto weights=Eigen::Map< Eigen::Vector<double, Eigen::Dynamic> >(fx.Data(),fx.Size());
    auto results=op->mult(weights);


    auto y_map=Eigen::Map< Eigen::Vector<double, Eigen::Dynamic> >(fy.Data(),fy.Size());
    y_map=results;
    //y *= 1.0 / (4*M_PI);
  }

  };

#endif  
  
}

#endif
#endif

