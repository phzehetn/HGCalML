# Needed Layers and model_blocks

* pre-snowflake
  * [x] condition_input
    * [x] Layer: CreateTruthSpectatorWeights
    * [x] Layer: CastRowSplits
    * [x] Layer: ProcessFeatures
    * [x] Layer: XYZtoXYZPrime
  * [x] double_tree_condensation_block
    * [x] Layer: KNN
      * [x] baseModule: LayerWithMetrics -> Should be fixed and removed
      * [x] slicing_knn_op: SlicingKnn
        * [x] oc_helper_ops: SelectWithDefault
        * [x] globals: knn_ops_use_tf_gradients
      * [x] binned_select_knn_op: BinnedSelectKnn
        * [x] bin_by_coordinates_op: BinByCoordinates
        * [x] index_replacer_op: IndexReplacer
        * [x] oc_helper_ops: SelectWithDefault
      * [x] oc_helper_ops: SelectWithDefault
    * [x] Layer: LocalDistanceScaling
      * [x] Layer: LayerWithMetrics
    * [x] GravNetLayersRagged: TranslationInvariantMP
      * [x] layernorm
      * [x] accknn_op -> AccumulateKnn
      * [x] SelectWithDefault
    * [x] model_block: tree_condensation_block
      * [x] model_block: GravNet_plus_TEQMP
      * [x] model_block: mini_tree_create
        * [x] LL: LLClusterCoordinates
          * [x] LL: LossLayerBase
            * [x] baseModule: LayerWithMetrics
          * [x] oc_helper_ops: CreateMidx
          * [x] SelectWithDefault
        * [x] LL: LLValuePenalty
          * [x] LL: LossLayerBase
        * [x] GCL: LLGraphCondensationScore
          * [x] LL: LossLayerBase
          * [x] LL: smooth_max
        * [x] GCL: CreateGraphCondensation
          * [x] GCL: GraphCondensation
          * [x] GCL: RestrictedDict
        * [x] GCL: MLGraphCondensationMetrics
          * [x] MetricsLayers: MLReductionMetrics
            * [x] MetricsLayer: MLBase
              * [x] LayerWithMetrics
            * [x] GravNetLayersRagged: SelectFromIndices
      * [x] model_block: mini_tree_clustering
        * [x] GCL: AddNeighbourDiff
          * [x] GCL: GraphCondensation
          * [x] GravNetLayersRagged: SelectFromIndices
        * [x] GCL: LLGraphCondensationEdges
          * [x] LL: LossLayerBase
        * [x] GCL: InsertEdgesIntoTransition
          * [x] GCL: GraphCondensation
        * [x] GCL: PushUp
          * [x] GCL: GraphCondensation
          * [x] push_knn_op: PushKNN
        * [x] GCL: SelectUp
    * [x] DebugLayer: PlotGraphCondensationEfficiency
      * [x] DebugLayer: _DebugPlotBase
      * [x] DebugLayer: CumulativeArray
    * [x] Layer: DummyLayer
    * [x] model_block: post_tree_condensation_push
      * [x] GCL: Mix
      * [x] GCL: PushUp
    * [x] model_block: tree_condensation_block2
      * [x] model_block: tree_condensation_block
        * [x] model_block: GravNet_plus_TEQMP
        * [x] mini_tree_create
        * [x] mini_tree_clustering
    * [x] Layer: SelectUp
  * [x] create_outputs
  * [x] GravNetLayersRagged ProcessFeatures
  * [x] DebugLayers: PlotCoordinates
    * [x] plotting_tools: shuffle_truth_colors
  * [x] ExtendedObjectCondensation5
    * [x] ExtendedObjectCondensation
      * [x] LLFullObjectCondensation
        * [x] LossLayerBase
        * [x] object_condensation Basic_OC_per_sample
        * [x] object_condensation PushPull_OC_per_sample
        * [x] object_condensation PreCond_OC_per_sample
        * [x] object_condensation Hinge_OC_per_sample_damped
        * [x] object_condensation Hinge_OC_per_sample
        * [x] object_condensation Hinge_Manhatten_OC_per_sample
        * [x] object_condensation Dead_Zone_Hinge_OC_per_sample
        * [x] object_condensation Hinge_OC_per_sample_learnable_qmin
        * [x] object_condensation Hinge_OC_per_sample_learnable_qmin_betascale_position
        * [x] object_condensation OC_loss
        * [x] object_condensation:
          * [x] binned_select_knn_op
          * [x] CreateMidx
          * [x] SelectWithDefault
        * [x] LossLayers: huber
        * [x] LossLayers: quantile
* full-model
  * [x] condition_input
  * [x] GravNet_plus_TEQMP
  * [x] create_outputs
  * [x] ProcessFeatures
  * [x] PlotCoordinates
  * [x] ExtendedObjectCondensation
  * [x] GravNetLayersRagged CastRowSplits
* [x] predict_hgcal -> HGcalPredictor
  * [x] Layers: LossLayerBase
* [x] analyse_hgcal_predictions
  * [x] OCHit2ShowersLayer
    * [x] assign_condensate_op BuildAndAssignCondensatesBinned
      * [x] bin_by_coordinates_op BinByCoordinates
      * [x] assign_condensate_op merge_noise_as_indiv_cp
      * [x] assign_condensate_op cumsum_ragged
  * [x] process_endcap -> process_endcap
    * [x] process_endcap -> OCGatherNumpy
  * [x] ShowersMatcher3 -> ShowersMatcher
    * [x] ShowersMatcher3 calculate_iou_serial_fast
      * [x] ShowersMatcher3 -> _findIdx
      * [x] ShowersMatcher3 -> calculate_iou_serial_fast_comp
  * [x] extra_plots
  * [x] visualize_events
