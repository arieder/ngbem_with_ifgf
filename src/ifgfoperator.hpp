#ifndef FILE_IFGFOPERATOR
#define FILE_IFGFOPERATOR


#include "fmmoperator.hpp"
#include "ngbem.hpp"


namespace ngbem
{

    template<typename KERNEL>
    class  IFGF_Operator : public FMM_Operator<KERNEL >
    {
    public:
	IFGF_Operator(KERNEL _kernel, Array<Vec<3> > _xpts, Array<Vec<3> > _ypts,
		      Array<Vec<3>> _xnv, Array<Vec<3>> _ynv,const BEMParameters& param)
	    :
	    FMM_Operator<KERNEL>(_kernel,std::move(_xpts), std::move( _ypts), std::move(_xnv), std::move(_ynv))
	{

	}
      
    };
#ifdef USE_IFGF

#include <helmholtz_ifgf.hpp>
#include <combined_field_helmholtz_ifgf.hpp>
#include <laplace_ifgf.hpp>
#include <Eigen/Dense>

    #include <cstdlib>
#include <tbb/task_arena.h>
#include <tbb/global_control.h>
#include <fenv.h>
#include <fstream>
    
  template<>
  class IFGF_Operator<HelmholtzSLKernel<3> > : public Base_FMM_Operator<std::complex<double> > 
  {
      typedef HelmholtzSLKernel<3>  KERNEL;
      typedef HelmholtzIfgfOperator<3> OperatorType;
      typedef Base_FMM_Operator<std::complex<double > > BASE;

  protected:
      std::unique_ptr<OperatorType> op;
      KERNEL kernel;

  public:
      IFGF_Operator(KERNEL _kernel, Array<Vec<3> > _xpts, Array<Vec<3> > _ypts,
		    Array<Vec<3>> _xnv, Array<Vec<3>> _ynv, const BEMParameters& param)
	  : BASE(std::move(_xpts), std::move( _ypts), std::move(_xnv), std::move(_ynv)),
	    kernel(_kernel)
      {
	  std::cout<<"creating ifgf opitty"<<std::endl;


	  size_t leafSize=param.leafsize;
	  size_t order=param.expansion_order;
	  int n_elem=param.n_elements;
	  double tol=param.eps;
	  double waveNumber=_kernel.GetKappa();

	  std::cout<<"size="<<xpts.Size()<<std::endl;
	  std::cout<<"size="<<ypts.Size()<<std::endl;


	  //auto global_control = tbb::global_control( tbb::global_control::max_allowed_parallelism,      12);                                                                                                                                                                                   
	  op=make_unique<HelmholtzIfgfOperator<3> > (waveNumber,leafSize,order,n_elem,tol);
	  
	  auto srcs=Eigen::Map<typename OperatorType::PointArray>( xpts[0].Data(),3, xpts.Size());
	  auto targets=Eigen::Map<typename OperatorType::PointArray>(ypts[0].Data(),3, ypts.Size());


/*	  const static Eigen::IOFormat CSVFormat(Eigen::FullPrecision, Eigen::DontAlignCols, ", ", "\n");
          std::ofstream file("srcs.csv");
          file<<srcs.format(CSVFormat);
          file.close();
*/	  
	  op->init(srcs,targets);	
      }


      void  Mult(const BaseVector & x, BaseVector & y) const 
      {
	  std::cout<<"ifgf mult"<<std::endl;
	  static Timer tall("ngbem fmm apply HelmholtzCF (IFGF)"); RegionTimer reg(tall);
	  auto fx = x.FV<Complex>();
	  auto fy = y.FV<Complex>();

		  //fy = 0;
	  //auto global_control = tbb::global_control( tbb::global_control::max_allowed_parallelism,      12);                                                                                                                                                                                   


	  auto weights=Eigen::Map< Eigen::Vector<std::complex<double>, Eigen::Dynamic> >(fx.Data(),fx.Size());
	  auto results=op->mult(weights);


	  auto y_map=Eigen::Map< Eigen::Vector<std::complex<double>, Eigen::Dynamic> >(fy.Data(),fy.Size());
	  y_map=results;
	  //y *= 1.0 / (4*M_PI);
      }

  };


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

