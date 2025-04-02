#ifndef __NEARFIELD_OPERATOR__
#define __NEARFIELD_OPERATOR__


namespace ngbem {

template <typename KERNEL>
class NearfieldOperator : public BaseMatrix
{
protected:
    KERNEL kernel;
    using value_type = typename  KERNEL::value_type;  
	
public:
    NearfieldOperator(KERNEL _kernel, const Table<int>& _nbels, shared_ptr<FESpace> _trial_space, shared_ptr<FESpace> _test_space, 
		      shared_ptr<DifferentialOperator> _trial_evaluator, 
		      shared_ptr<DifferentialOperator> _test_evaluator,
		      int _intOrder) :
	kernel(_kernel),
	nbels((_nbels)),
	trial_space(_trial_space),
	test_space(_test_space),
	trial_evaluator(_trial_evaluator),
	test_evaluator(_test_evaluator),
	intOrder(_intOrder)
	{


	    tie(identic_panel_x, identic_panel_y, identic_panel_weight) = IdenticPanelIntegrationRule(intOrder);
	    tie(common_vertex_x, common_vertex_y, common_vertex_weight) = CommonVertexIntegrationRule(intOrder);
	    tie(common_edge_x, common_edge_y, common_edge_weight) = CommonEdgeIntegrationRule(intOrder);
	    
	}
    void Mult(const BaseVector & x, BaseVector & y) const override
	{
	    std::cout<<"NF mult"<<intOrder<<std::endl;	   
	    LocalHeap lh(100000000);
	    IntegrationRule ir(ET_TRIG, intOrder);
	    auto trial_mesh = trial_space->GetMeshAccess();
	    auto test_mesh = test_space->GetMeshAccess();
	    
	    trial_mesh->IterateElements
		 (BND, lh, [&](auto ei_trial, LocalHeap & lh)
		      {
			   for (auto nrtest : nbels[ei_trial.Nr()])
			   {
				ElementId ei_test(BND, nrtest);
            
				auto & trial_trafo = trial_mesh -> GetTrafo(ei_trial, lh);
				auto & test_trafo = test_mesh -> GetTrafo(ei_test, lh);
				auto & trial_fel = trial_space->GetFE(ei_trial, lh);
				auto & test_fel = test_space->GetFE(ei_test, lh);
            
				Array<DofId> trial_dnums(trial_fel.GetNDof(), lh);
				Array<DofId> test_dnums(test_fel.GetNDof(), lh);
            
				trial_space->GetDofNrs (ei_trial, trial_dnums);        
				test_space->GetDofNrs (ei_test, test_dnums);
            
				FlatMatrix<value_type> elmat(test_dnums.Size(), trial_dnums.Size(), lh);
				//tassSS.Start();				
				CalcElementMatrix (elmat, ei_trial, ei_test, lh);
				//tassSS.Stop();
				//tasscorr.Start();        
            
				// subtract terms from fmm:
            
				MappedIntegrationRule<2,3> trial_mir(ir, trial_trafo, lh);
				MappedIntegrationRule<2,3> test_mir(ir, test_trafo, lh);
            
				FlatMatrix<> shapesi(test_fel.GetNDof(), test_evaluator->Dim()*ir.Size(), lh);
				FlatMatrix<> shapesj(trial_fel.GetNDof(), trial_evaluator->Dim()*ir.Size(), lh);
            
				test_evaluator -> CalcMatrix(test_fel, test_mir, Trans(shapesi), lh);
				trial_evaluator-> CalcMatrix(trial_fel, trial_mir, Trans(shapesj), lh);
            
				for (auto term : kernel.terms)
				{
				     HeapReset hr(lh);
				     FlatMatrix<value_type> kernel_ixiy(ir.Size(), ir.Size(), lh);
				     for (int ix = 0; ix < ir.Size(); ix++)
				     {
					  for (int iy = 0; iy < ir.Size(); iy++)
					  {
					       Vec<3> x = test_mir[ix].GetPoint();
					       Vec<3> y = trial_mir[iy].GetPoint();
                        
					       Vec<3> nx = test_mir[ix].GetNV();
					       Vec<3> ny = trial_mir[iy].GetNV();
					       value_type kernel_ = 0.0;
					       if (L2Norm2(x-y) > 0)
						    kernel_ = kernel.Evaluate(x, y, nx, ny)(term.kernel_comp);
                        
					       double fac = test_mir[ix].GetWeight()*trial_mir[iy].GetWeight();
					       kernel_ixiy(ix, iy) = term.fac*fac*kernel_;
					  }
				     }
                
				     FlatMatrix<value_type> kernel_shapesj(ir.Size(), trial_fel.GetNDof(), lh);
				     FlatMatrix<> shapesi1(test_fel.GetNDof(), ir.Size(), lh);
				     FlatMatrix<> shapesj1(trial_fel.GetNDof(), ir.Size(), lh);
                
				     for (int j = 0; j < ir.Size(); j++)
				     {
					  shapesi1.Col(j) = shapesi.Col(test_evaluator->Dim()*j+term.test_comp);
					  shapesj1.Col(j) = shapesj.Col(trial_evaluator->Dim()*j+term.trial_comp);
				     }
				     kernel_shapesj = kernel_ixiy * Trans(shapesj1);
				     elmat -= shapesi1 * kernel_shapesj;
				}
				//tasscorr.Stop();        
				//nearfield_correction -> AddElementMatrix (test_dnums, trial_dnums, elmat);

				FlatVector<typename KERNEL::value_type> xv(trial_dnums.Size(),lh);
				x.GetIndirect(trial_dnums,xv);			
				FlatVector<typename KERNEL::value_type> yv(test_dnums.Size(),lh);

				yv=(elmat)*xv;								
				y.AddIndirect(test_dnums,yv,true);

	    

			   }
		      });

	    
	}


        void CalcElementMatrix(
            FlatMatrix<value_type> matrix, ElementId ei_trial,
            ElementId ei_test, LocalHeap &lh) const {
            auto mesh = this->trial_space->GetMeshAccess();
            auto mesh2 = this->test_space->GetMeshAccess();

            static Timer tall("ngbem - elementmatrix " + KERNEL::Name());
            RegionTimer reg(tall);

            static Timer t1("ngbem - elementmatrix, part1  " + KERNEL::Name());

            t1.Start();
            IntegrationRule irtrig(ET_TRIG, intOrder);
            /*
            auto [ identic_panel_x, identic_panel_y, identic_panel_weight ] =
              IdenticPanelIntegrationRule(param.intorder);

            auto [ common_vertex_x, common_vertex_y, common_vertex_weight ] =
              CommonVertexIntegrationRule(param.intorder);

            auto [ common_edge_x, common_edge_y, common_edge_weight ] =
              CommonEdgeIntegrationRule(param.intorder);
            */
            matrix = 0;
            t1.Stop();

            Vec<3> x, y, nx, ny;
            typedef decltype(kernel.Evaluate(x, y, nx, ny)) KERNEL_COMPS_T;

            // common code for same panel, common edge, common vertex
            auto Integrate4D = [&](const IntegrationRule &irx,
                                   const IntegrationRule &iry,
                                   const FiniteElement &feli,
                                   const FiniteElement &felj,
                                   const ElementTransformation &trafoi,
                                   const ElementTransformation &trafoj,
                                   FlatMatrix<value_type> elmat,
                                   LocalHeap &lh) {
              HeapReset hr(lh);
              SIMD_IntegrationRule simd_irx(irx);
              SIMD_IntegrationRule simd_iry(iry);

              SIMD_MappedIntegrationRule<2, 3> mirx(simd_irx, trafoi, lh);
              SIMD_MappedIntegrationRule<2, 3> miry(simd_iry, trafoj, lh);

              FlatMatrix<SIMD<double>> mshapesi(
                  feli.GetNDof() * test_evaluator->Dim(), mirx.Size(), lh);
              FlatMatrix<SIMD<value_type>> mshapesi_kern(feli.GetNDof(),
                                                         mirx.Size(), lh);
              FlatMatrix<SIMD<double>> mshapesj(
                  felj.GetNDof() * trial_evaluator->Dim(), miry.Size(), lh);

              test_evaluator->CalcMatrix(feli, mirx, mshapesi);
              trial_evaluator->CalcMatrix(felj, miry, mshapesj);

              FlatVector<Vec<KERNEL_COMPS_T::SIZE, SIMD<value_type>>>
                  kernel_values(mirx.Size(), lh);
              for (int k2 = 0; k2 < mirx.Size(); k2++) {
                Vec<3, SIMD<double>> x = mirx[k2].Point();
                Vec<3, SIMD<double>> y = miry[k2].Point();
                Vec<3, SIMD<double>> nx = mirx[k2].GetNV();
                Vec<3, SIMD<double>> ny = miry[k2].GetNV();
                kernel_values(k2) =
                    mirx[k2].GetMeasure() * miry[k2].GetMeasure() *
                    simd_irx[k2].Weight() * kernel.Evaluate(x, y, nx, ny);
              }
              for (auto term : kernel.terms) {
                auto mshapesi_comp =
                    mshapesi.RowSlice(term.test_comp, test_evaluator->Dim());
                for (int k2 = 0; k2 < mirx.Size(); k2++) {
                  SIMD<value_type> kernel_ =
                      kernel_values(k2)(term.kernel_comp);
                  mshapesi_kern.Col(k2) =
                      term.fac * kernel_ * mshapesi_comp.Col(k2);
                }

                AddABt(
                    mshapesi_kern,
                    mshapesj.RowSlice(term.trial_comp, trial_evaluator->Dim())
                        .AddSize(felj.GetNDof(), miry.Size()),
                    elmat);
              }
            };

            auto verti = mesh2->GetElement(ei_test).Vertices();
            auto vertj = mesh->GetElement(ei_trial).Vertices();

            FiniteElement &feli = test_space->GetFE(ei_test, lh);
            FiniteElement &felj = trial_space->GetFE(ei_trial, lh);

            ElementTransformation &trafoi = mesh2->GetTrafo(ei_test, lh);
            ElementTransformation &trafoj = mesh->GetTrafo(ei_trial, lh);

            Array<DofId> dnumsi, dnumsj;
            test_space->GetDofNrs(ei_test, dnumsi); // mapping to global dof
            trial_space->GetDofNrs(ei_trial, dnumsj);

            FlatMatrix<> shapei(feli.GetNDof(), test_evaluator->Dim(), lh);
            FlatMatrix<> shapej(felj.GetNDof(), trial_evaluator->Dim(), lh);

            matrix = 0.0;

            int n_common_vertices = 0;
            for (auto vi : verti)
              if (vertj.Contains(vi))
                n_common_vertices++;

            switch (n_common_vertices) {
            case 3: // identical panel
            {
              // RegionTimer reg(t_identic);

              constexpr int BS = 128;
              for (int k = 0; k < identic_panel_weight.Size(); k += BS) {
                int num = std::min(size_t(BS), identic_panel_weight.Size() - k);

                HeapReset hr(lh);

                IntegrationRule irx(num, lh);
                IntegrationRule iry(num, lh);

                for (int k2 = 0; k2 < num; k2++) {
                  Vec<2> xk = identic_panel_x[k + k2];
                  Vec<2> yk = identic_panel_y[k + k2];

                  irx[k2] = IntegrationPoint(xk(0), xk(1), 0,
                                             identic_panel_weight[k + k2]);
                  iry[k2] = IntegrationPoint(yk(0), yk(1), 0, 0);
                }

                Integrate4D(irx, iry, feli, felj, trafoi, trafoj, matrix, lh);
              }

              break;
            }
            case 2: // common edge
            {
              // RegionTimer reg(t_common_edge);

              const EDGE *edges =
                  ElementTopology::GetEdges(ET_TRIG); // 0 1 | 1 2 | 2 0
              int cex, cey;
              for (int cx = 0; cx < 3; cx++)
                for (int cy = 0; cy < 3; cy++) {
                  IVec<2> ex(verti[edges[cx][0]], verti[edges[cx][1]]);
                  IVec<2> ey(vertj[edges[cy][0]], vertj[edges[cy][1]]);
                  if (ex.Sort() == ey.Sort()) {
                    cex = cx; // -> "common" edge number triangle i
                    cey = cy; // -> "common" edge number triangle j
                    break;
                  }
                }
              int vpermx[3] = {edges[cex][0], edges[cex][1],
                               -1}; // common edge gets first
              vpermx[2] = 3 - vpermx[0] - vpermx[1];
              int vpermy[3] = {edges[cey][1], edges[cey][0],
                               -1}; // common edge gets first
              vpermy[2] = 3 - vpermy[0] - vpermy[1];

              constexpr int BS = 128;
              for (int k = 0; k < common_edge_weight.Size(); k += BS) {
                int num = std::min(size_t(BS), common_edge_weight.Size() - k);

                HeapReset hr(lh);

                IntegrationRule irx(num, lh);
                IntegrationRule iry(num, lh);

                for (int k2 = 0; k2 < num; k2++) {
                  Vec<2> xk = common_edge_x[k + k2];
                  Vec<2> yk = common_edge_y[k + k2];

                  Vec<3> lamx(1 - xk(0) - xk(1), xk(0), xk(1));
                  Vec<3> lamy(1 - yk(0) - yk(1), yk(0), yk(1));

                  Vec<3> plamx, plamy;
                  for (int i = 0; i < 3; i++) {
                    plamx(vpermx[i]) = lamx(i);
                    plamy(vpermy[i]) = lamy(i);
                  }

                  irx[k2] = IntegrationPoint(plamx(0), plamx(1), 0,
                                             common_edge_weight[k + k2]);
                  iry[k2] = IntegrationPoint(plamy(0), plamy(1), 0, 0);
                }

                Integrate4D(irx, iry, feli, felj, trafoi, trafoj, matrix, lh);
              }

              break;
            }

            case 1: // common vertex
            {
              // RegionTimer reg(t_common_vertex);

              int cvx = -1, cvy = -1;
              for (int cx = 0; cx < 3; cx++)
                for (int cy = 0; cy < 3; cy++) {
                  if (verti[cx] == vertj[cy]) {
                    cvx = cx;
                    cvy = cy;
                    break;
                  }
                }

              int vpermx[3] = {cvx, (cvx + 1) % 3, (cvx + 2) % 3};
              vpermx[2] = 3 - vpermx[0] - vpermx[1];
              int vpermy[3] = {cvy, (cvy + 1) % 3, (cvy + 2) % 3};
              vpermy[2] = 3 - vpermy[0] - vpermy[1];

              // vectorized version:
              constexpr int BS = 128;
              for (int k = 0; k < common_vertex_weight.Size(); k += BS) {
                int num = std::min(size_t(BS), common_vertex_weight.Size() - k);

                HeapReset hr(lh);

                IntegrationRule irx(num, lh);
                IntegrationRule iry(num, lh);

                for (int k2 = 0; k2 < num; k2++) {
                  Vec<2> xk = common_vertex_x[k + k2];
                  Vec<2> yk = common_vertex_y[k + k2];

                  Vec<3> lamx(1 - xk(0) - xk(1), xk(0), xk(1));
                  Vec<3> lamy(1 - yk(0) - yk(1), yk(0), yk(1));

                  Vec<3> plamx, plamy;
                  for (int i = 0; i < 3; i++) {
                    plamx(vpermx[i]) = lamx(i);
                    plamy(vpermy[i]) = lamy(i);
                  }

                  irx[k2] = IntegrationPoint(plamx(0), plamx(1), 0,
                                             common_vertex_weight[k + k2]);
                  iry[k2] = IntegrationPoint(plamy(0), plamy(1), 0, 0);
                }

                Integrate4D(irx, iry, feli, felj, trafoi, trafoj, matrix, lh);
              }

              break;
            }

            case 0: // disjoint panels
            {
              // RegionTimer r(t_disjoint);

              // shapes+geom out of loop, matrix multiplication
              MappedIntegrationRule<2, 3> mirx(irtrig, trafoi, lh);
              MappedIntegrationRule<2, 3> miry(irtrig, trafoj, lh);

              FlatMatrix<> shapesi(feli.GetNDof(),
                                   test_evaluator->Dim() * irtrig.Size(), lh);
              FlatMatrix<> shapesj(felj.GetNDof(),
                                   trial_evaluator->Dim() * irtrig.Size(), lh);

              test_evaluator->CalcMatrix(feli, mirx, Trans(shapesi), lh);
              trial_evaluator->CalcMatrix(felj, miry, Trans(shapesj), lh);

              for (auto term : kernel.terms) {
                HeapReset hr(lh);
                FlatMatrix<value_type> kernel_ixiy(irtrig.Size(), irtrig.Size(),
                                                   lh);
                for (int ix = 0; ix < irtrig.Size(); ix++) {
                  for (int iy = 0; iy < irtrig.Size(); iy++) {
                    Vec<3> x = mirx[ix].GetPoint();
                    Vec<3> y = miry[iy].GetPoint();

                    Vec<3> nx = mirx[ix].GetNV();
                    Vec<3> ny = miry[iy].GetNV();
                    value_type kernel_ =
                        kernel.Evaluate(x, y, nx, ny)(term.kernel_comp);

                    double fac = mirx[ix].GetWeight() * miry[iy].GetWeight();
                    kernel_ixiy(ix, iy) = term.fac * fac * kernel_;
                  }
                }

                FlatMatrix<value_type> kernel_shapesj(irtrig.Size(),
                                                      felj.GetNDof(), lh);
                FlatMatrix<> shapesi1(feli.GetNDof(), irtrig.Size(), lh);
                FlatMatrix<> shapesj1(felj.GetNDof(), irtrig.Size(), lh);

                for (int j = 0; j < irtrig.Size(); j++) {
                  shapesi1.Col(j) =
                      shapesi.Col(test_evaluator->Dim() * j + term.test_comp);
                  shapesj1.Col(j) =
                      shapesj.Col(trial_evaluator->Dim() * j + term.trial_comp);
                }
                kernel_shapesj = kernel_ixiy * Trans(shapesj1);
                matrix += shapesi1 * kernel_shapesj;
              }
              break;
            }
            default:
              throw Exception("not possible2 ");
            }
        }

        AutoVector CreateRowVector () const override
	{
	    return make_unique<VVector<typename KERNEL::value_type>>(test_space->GetNDof());
	}
	AutoVector CreateColVector () const override
	{
	    return make_unique<VVector<typename KERNEL::value_type>>(trial_space->GetNDof());
	}



    private:
	Table<int> nbels;
	shared_ptr<FESpace> trial_space;
	shared_ptr<FESpace> test_space;
	shared_ptr<DifferentialOperator> trial_evaluator;
	shared_ptr<DifferentialOperator> test_evaluator;


        Array<Vec<2>> identic_panel_x, identic_panel_y;
        Array<double> identic_panel_weight;
    
        Array<Vec<2>> common_vertex_x, common_vertex_y;
        Array<double> common_vertex_weight;

        Array<Vec<2>> common_edge_x, common_edge_y;
        Array<double> common_edge_weight;


	int intOrder;

    };

};






#endif
