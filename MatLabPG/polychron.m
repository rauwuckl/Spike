% Finds polychronous groups in the workspace generated by spnet.m
% and saved in the matlab file fname, e.g., fname = '3600.mat';
% Created by Eugene M.Izhikevich.           November 21, 2005
% Modified on April 2, 2008 based on suggestions of Petra Vertes (UK).

% Main idea: for each mother neuron, consider various combinations of
% pre-synatic (anchor) neurons and see whether any activity of a silent
% network could emerge if these anchors are fired.
clear;
clearvars -global


timestepMode=0;
model_izhikevich = 1;
model_conductanceLIAF = 2;
neuronModel = model_conductanceLIAF;

experimentName = '1.5--FF_FB_LAT_stdp_0.005_nCon2'


biological_conductance_scaling_constant_lambda_G2E_FF = 0.00002;
biological_conductance_scaling_constant_lambda_E2E_FF = 0.0001;
biological_conductance_scaling_constant_lambda_E2I_L = 0.002;
biological_conductance_scaling_constant_lambda_I2E_L = 0.004;
biological_conductance_scaling_constant_lambda_E2E_L = 0.00005;
%biological_conductance_scaling_constant_lambda_E2E_FB = 0.00001;



for trainedNet = [1]
    if (trainedNet)
        networkStatesName = 'networkStates_trained.mat'
    else
        networkStatesName = 'networkStates_untrained.mat'
    end

    initOn = 1;
    global a b c d N D pp s ppre dpre post pre delay T timestep di nLayers ExcitDim InhibDim
    if exist(['../output/' experimentName '/' networkStatesName],'file')
        load(['../output/' experimentName '/' networkStatesName]);
        disp('**ATN** network state data previously stored are loaded')
    end

    
    
    

    
    %%% parameters %%%
    % val from the simulation
    %M = 50; %number of syn per neurons
    ExcitDim = 64;
    InhibDim = 32;
    nLayers = 4;
    N = (ExcitDim*ExcitDim+InhibDim*InhibDim)*nLayers;% N: num neurons
    timestep = 0.00002;
    targetInputLayer = nLayers-1;
    
    
    
    D = 10; % D: max delay
    T= 100;%100;              % the max length of a group to be considered;
    if(timestepMode)
        T=5000;%1000;
        D = int32(D/(timestep*1000));
    end
    % parameters to be provided
    anchor_width=3;     % the number of anchor neurons, from which a group starts
%     min_group_path=2;%4;%7;   % discard all groups having shorter paths from the anchor neurons
    max_num_strong_anchors = 10;
    min_group_path = 2;
    min_group_path_to_plot = 2;
    min_multi_fanin_syns_to_plot = 7;
    
    saveImages = 0;
    plotFigure = 0;
    printDebugMessage = 1;

    %%% initial load of data %%%
    if exist(['../output/' experimentName '/' networkStatesName],'file')==0
        
        % loading data %
        disp('**ATN** loading data')


        if(trainedNet)
            fileID = fopen(['../output/' experimentName '/Neurons_NetworkWeights.bin']);
        else
            fileID = fopen(['../output/' experimentName '/Neurons_NetworkWeights_Initial.bin']);
        end
        weights_loaded = fread(fileID,'float32');
        fclose(fileID);

        fileID = fopen(['../output/' experimentName '/Neurons_NetworkDelays.bin']);
        delays_loaded = fread(fileID,'int32');
        fclose(fileID);

        fileID = fopen(['../output/' experimentName '/Neurons_NetworkPre.bin']);
        preIDs_loaded = fread(fileID,'int32');
        fclose(fileID);

        fileID = fopen(['../output/' experimentName '/Neurons_NetworkPost.bin']);
        postIDs_loaded = fread(fileID,'int32');
        fclose(fileID);
        
        
        cond = find(preIDs_loaded>=0);
        preIDs_loaded = preIDs_loaded(cond)+1; %index start from 1 in matlab
        postIDs_loaded = postIDs_loaded(cond)+1;
        weights_loaded = weights_loaded(cond);
        delays_loaded = delays_loaded(cond);
        
        if(timestepMode==0)
            delays_loaded = int32(delays_loaded*timestep*1000);
        end
        
        cond1 = mod(postIDs_loaded,(ExcitDim*ExcitDim+InhibDim*InhibDim))<=ExcitDim*ExcitDim;
        cond2 = postIDs_loaded > preIDs_loaded;
        FFWeights = weights_loaded(find(cond2==1 & cond1==1));
        figure;
        hist(FFWeights);
        title('Feed Forward Weight Distributions');
        
        %uncomment the command below to test the algorithm for shuffled (randomized e->e) synapses
        %e2e = find(s>=0 & post<Ne); s(e2e) = s(e2e(randperm(length(e2e))));

        groups={};          % the list of all polychronous groups



        % Make necessary initializations to speed-up simulations.

        %find max numPostSynCon:
        maxNumPostSynCon = 0;
        for i=1:N
            postListLen = length(find(preIDs_loaded==i));
            if maxNumPostSynCon<postListLen
                maxNumPostSynCon = postListLen;
            end
        end;

        post = zeros(N,maxNumPostSynCon);
        delay = zeros(N,maxNumPostSynCon)+1;
        s = zeros(N,maxNumPostSynCon);

        %ppre and pre contain lists of presynaptic ids connected to post synaptic
        %cell i
        pre = cell(1,N);%stores index of PreSynapse (nCells x nPreSynCons)
        ppre = cell(1,N);
        dpre = cell(1,N);
        pp = cell(1,N);

        for i_pre=1:N
            cond = preIDs_loaded == i_pre;

            delays_tmp = delays_loaded(cond);
            delay(i_pre,1:length(delays_tmp))=delays_tmp;
            % This matrix provides the delay values for each synapse.
            %delay values of synapses that are connected from each presynaptic id i;
            % ie delay{i} return a list of delays of synapses that is connected from i

            post_tmp = postIDs_loaded(cond);
            post(i_pre,1:length(post_tmp))=post_tmp;
            %post synaptic ids that are connected from each presynaptic id i;
            %ie post{i} return a list of ids of synapses that is connected from i

            for post_id_index = 1:length(post_tmp)
                i_post=post_tmp(post_id_index);
                pre{i_post}(length(pre{i_post})+1,1)=(i_pre+N*(post_id_index-1));
                ppre{i_post}(length(ppre{i_post})+1,1)=i_pre;
            end

            s_tmp = weights_loaded(cond);
            s(i_pre,1:length(s_tmp))=s_tmp;
        end


        % %This cell element tells what the presynaptic delay is;
        for i_post=1:N
            dpre{i_post}=delay( pre{i_post} );
        end;

        %This cell element tells where to put PSPs in the matrix I (N by 1000)
        for i_post=1:N
%             pp{i_post}=post(i_post,:)+N*(delay(i_post,:)-1);
            pp{i_post}=post(i_post,:)+N*(delay(i_post,:));
        end;

        %plotDistributions(ppre,nLayers,ExcitDim,InhibDim);
        
        save(['../output/' experimentName '/' networkStatesName]);
    end

    %plotDistributions(ppre,nLayers,ExcitDim,InhibDim);
    
    
    
    
    
    
    
    
    
    
    %%% filter synaptic weights %%%
    if (neuronModel == model_conductanceLIAF)
        
        sm = 0.0001 * 0.0008 %0.0008; %is the biological synaptic constant E2E_FF
        sm_threshold = 0.8*sm %0.8*sm;
%         sm_threshold_input = 0.95*sm;
        E2I_const = 10.0;
        E2E_const = 1.0;
        
        applyThForE2IToo = 0
        
        
        sorted = sort(s((ExcitDim*ExcitDim+InhibDim*InhibDim)*(targetInputLayer-1)+1:(ExcitDim*ExcitDim+InhibDim*InhibDim)*(targetInputLayer-1)+ExcitDim*ExcitDim));
        sm_threshold_input = sorted(int32(ExcitDim*ExcitDim*0.95))*sm;%take only 5% of the input with strong weight
        
        s = s*sm;
        if (applyThForE2IToo==1)
            s(find(s>0 & s<=sm_threshold))=0;%remove small vals
        end
        
        di = s;
        reversal_potential_Vhat_excit = 0.0;
        reversal_potential_Vhat_inhib = -0.074;
        for l = 1:nLayers
            excit_begin = (ExcitDim*ExcitDim+InhibDim*InhibDim)*(l-1)+1;
            excit_end = ExcitDim*ExcitDim*l + (InhibDim*InhibDim)*(l-1);
            inhib_begin = ExcitDim*ExcitDim*l + (InhibDim*InhibDim)*(l-1) + 1;
            inhib_end = (ExcitDim*ExcitDim+InhibDim*InhibDim)*l;
            
            s_tmp = s(excit_begin:excit_end,:);
            %%remove any E2E syn smaller than sm_threshold
            if(applyThForE2IToo==0)
%                 s_tmp = s(excit_begin:excit_end,:);
                cond1 = find(inhib_end <= post(excit_begin:excit_end,:) & s(excit_begin:excit_end,:)<=sm_threshold); %input from the same layer
                s_tmp(cond1)=0.0;
            end
            
            %% apply biological scaling constant for the E2E_L
            cond = find(excit_begin <= post(excit_begin:excit_end,:) & post(excit_begin:excit_end,:) <= excit_end); %input from the same layer
            s_tmp(cond)=E2E_const.*s_tmp(cond);
            s(excit_begin:excit_end,:) = s_tmp;
 
            di(excit_begin:excit_end,:)=s(excit_begin:excit_end,:)*reversal_potential_Vhat_excit;
 
            

            s(inhib_begin:inhib_end,:)=s(inhib_begin:inhib_end,:)*E2I_const;
            di(inhib_begin:inhib_end,:)=s(inhib_begin:inhib_end,:)*reversal_potential_Vhat_inhib;
        end
        
        
    elseif (neuronModel == model_izhikevich)
        
        % izhikevich model params:
        a = zeros(N,1)+0.1;%0.02; %decay rate [0.02, 0.1]
        b = 0.2; %sensitivity [0.25, 0.2]
        c = -65;%reset [-65,-55,-50] potential after spike
        d = zeros(N,1)+8; %reset [2,4,8]
        %http://www.izhikevich.org/publications/spikes.pdf
        
        sm = 17.0;%18 ;%max synaptic weight
        sm_threshold = 0.8*sm;
        sm_threshold_input = 0.80*sm;
        E_IsynEfficRatio = 1.0;
    
        % multiply the weight by a constance used in the original analysis
        s = s*sm;
        %remove small values for the synaptic weights:
        s(find(s>0 & s<=sm_threshold))=0;%remove small vals
        % set inhibitory synaptic weight negative
        for l = 1:nLayers 
            i_begin = ExcitDim*ExcitDim*l + (InhibDim*InhibDim)*(l-1) + 1;
            i_end = (ExcitDim*ExcitDim+InhibDim*InhibDim)*l;
            s(i_begin:i_end,:)=s(i_begin:i_end,:)*-1*E_IsynEfficRatio;
        end
    end

    
    

    
    
    
%     if(trainedNet)
% %         load(['../output/' experimentName '/groups_trained.mat']);
%         fileID_id = fopen(['../output/' experimentName '/Neurons_SpikeIDs_Trained_Epoch0.bin']);
%         fileID_time = fopen(['../output/' experimentName '/Neurons_SpikeTimes_Trained_Epoch0.bin']);
%     else
% %         load(['../output/' experimentName '/groups_untrained.mat']);
%         fileID_id = fopen(['../output/' experimentName '/Neurons_SpikeIDs_Untrained_Epoch0.bin']);
%         fileID_time = fopen(['../output/' experimentName '/Neurons_SpikeTimes_Untrained_Epoch0.bin']);
%     end
%     
%     spikes_id = fread(fileID_id,'int32');
%     fclose(fileID_id);
%     
%     spikes_time = fread(fileID_time,'float32');%second
%     fclose(fileID_time);
%     
    
    
    
    
    
    %%% begin analysis %%%
    iterationCount = 0;
    if (plotFigure)
        fig = figure('position', [0, 0, 2000, 1500]);
    end
    for i=1:(ExcitDim*ExcitDim)
%     for i=754:(ExcitDim*ExcitDim)
        x = mod(i-1,ExcitDim)+1;
        y = floor(i/ExcitDim);
%         if ~(ExcitDim/4<=x && x<ExcitDim*3/4 && ExcitDim/4<=y && y<ExcitDim*3/4)
%             continue;
%         end
        iterationCount=iterationCount+1;
        if (printDebugMessage)
%             disp([num2str(round(iterationCount*100/(ExcitDim*ExcitDim/4))) '% i=' num2str(i) ' (' num2str(x) ', ' num2str(y) ')']);
            disp([num2str(round(iterationCount*100/(ExcitDim*ExcitDim))) '% i=' num2str(i) ' (' num2str(x) ', ' num2str(y) ')']);

        end
        i_post = i + (ExcitDim*ExcitDim + InhibDim*InhibDim)*targetInputLayer;%looking at the second layer
        anchors=1:anchor_width;                     % initial choice of anchor neurons

        pre_cells_FF = pre{i_post}(find(ppre{i_post}<=(ExcitDim*ExcitDim)+(ExcitDim*ExcitDim + InhibDim*InhibDim)*(targetInputLayer-1))); %to exlude input from lateral connections

        
        
        [B,sortedIndex] = sort(s(pre_cells_FF),'descend');
        strong_pre_id_to_specific_postCell=find(s(pre_cells_FF)>=sm_threshold_input);
        
        
    %     strong_pre=find(s(pre{i_post})>sm_threshold);    % list of the indecies of candidates for anchor neurons
        if length(strong_pre_id_to_specific_postCell) >= anchor_width       % must be enough candidates
            if length(strong_pre_id_to_specific_postCell) >= max_num_strong_anchors
                strong_pre_id_to_specific_postCell = sortedIndex(1:max_num_strong_anchors);
            end
            if (printDebugMessage)
                disp(num2str(reshape(strong_pre_id_to_specific_postCell,1,length(strong_pre_id_to_specific_postCell))));
            end
            while 1        % will get out of the loop via the 'break' command below

                if(length(unique(ppre{i_post}(strong_pre_id_to_specific_postCell(anchors))))==anchor_width)%skip if the set contains duplicate presynaptic cells
                
                    if (printDebugMessage)
                        disp([num2str(round(iterationCount*100/(ExcitDim*ExcitDim))) '%  anchors: ' num2str(reshape(strong_pre_id_to_specific_postCell(anchors),1,anchor_width)) ' (' num2str(length(strong_pre_id_to_specific_postCell)) ')']);
                    end

                    maxDelay = max(dpre{i_post}(strong_pre_id_to_specific_postCell(anchors)))+1;
                    gr=polygroup( ppre{i_post}(strong_pre_id_to_specific_postCell(anchors)), maxDelay-dpre{i_post}(strong_pre_id_to_specific_postCell(anchors)),neuronModel );

                    %Calculate the longest path from the first to the last spike
                    fired_path=sparse(N,1);        % the path length of the firing (from the anchor neurons)
                    for j=1:length(gr.gr(:,2))
                        fired_path( gr.gr(j,4) ) = max( fired_path( gr.gr(j,4) ), 1+fired_path( gr.gr(j,2) ));
                    end;
                    longest_path = max(fired_path);

                    
                    if longest_path>=min_group_path
%                     if size(gr.firings,1)>anchor_width+1

                        gr.longest_path = longest_path(1,1); % the path is a cell

                        % How many times were the spikes from the anchor neurons useful?
                        % (sometimes an anchor neuron does not participate in any
                        % firing, because the mother neuron does its job; such groups
                        % should be excluded. They are found when the mother neuron is
                        % an anchor neuron for some other neuron).
                        useful = zeros(1,anchor_width);
                        anch = ppre{i_post}(strong_pre_id_to_specific_postCell(anchors));
                        for j=1:anchor_width
                            useful(j) = length( find(gr.gr(:,2) == anch(j) ) );
                        end;
                        if all(useful>=2)
        %                     ppre{i_post}(strong_pre(anchors))
                            groups{end+1}=gr;           % add found group to the list
                            disp(['   groups=' num2str(c) ', size=' num2str(size(gr.firings,1)) ', path_length=' num2str(gr.longest_path)])   % display of the current status
%                             if (plotFigure && longest_path>=min_group_path_to_plot)
                            if (plotFigure && length( find(gr.gr(:,6) == 2 ) )>min_multi_fanin_syns_to_plot && longest_path>=min_group_path_to_plot)
                                
                                if(timestepMode)%calculate based on timestep but plot as ms
                                    plot(gr.firings(:,1)*timestep*1000,gr.firings(:,2),'o');
                                    strValues = strtrim(cellstr(num2str([gr.firings(:,2), gr.firings(:,1)*timestep*1000],'(%d,%3.1f)')));
                                    text(gr.firings(:,1)*timestep*1000,gr.firings(:,2),strValues,'VerticalAlignment','bottom');
                                    hold on;
    %                                 plot(gr.firings_inhib(:,1),gr.firings_inhib(:,2),'.', 'MarkerSize',15);
                                else 
                                    plot(gr.firings(:,1),gr.firings(:,2),'o');
                                    strValues = strtrim(cellstr(num2str([gr.firings(:,2), gr.firings(:,1)],'(%d, %3.1f)')));
                                    text(gr.firings(:,1),gr.firings(:,2),strValues,'VerticalAlignment','bottom');
                                    hold on;
    %                                 plot(gr.firings_inhib(:,1)*timestep*1000,gr.firings_inhib(:,2),'.', 'MarkerSize',15);

                                end
                                for l=1:nLayers
                                    if(timestepMode)
                                        plot([0 T*timestep*1000],[(ExcitDim*ExcitDim+InhibDim*InhibDim)*l (ExcitDim*ExcitDim+InhibDim*InhibDim)*l],'k');
                                        plot([0 T*timestep*1000],[(ExcitDim*ExcitDim)*l+(InhibDim*InhibDim)*(l-1) (ExcitDim*ExcitDim)*l+(InhibDim*InhibDim)*(l-1)],'k--');
                                    else
                                        plot([0 T],[(ExcitDim*ExcitDim+InhibDim*InhibDim)*l (ExcitDim*ExcitDim+InhibDim*InhibDim)*l],'k');
                                        plot([0 T],[(ExcitDim*ExcitDim)*l+(InhibDim*InhibDim)*(l-1) (ExcitDim*ExcitDim)*l+(InhibDim*InhibDim)*(l-1)],'k--');
                                    end
                                end
                                for j=1:size(gr.gr,1)
                                    if gr.gr(j,6)==1
                                        lineCol = [0.5 0.5 0.5];
                                    else
                                        lineCol = [0,0,0];
                                    end
                                    if(timestepMode)
                                        plot(gr.gr(j,[1 3 5])*timestep*1000,gr.gr(j,[2 4 4]),'.-','LineWidth',gr.gr(j,6),'color',lineCol);
                                    else
                                        plot(gr.gr(j,[1 3 5]),gr.gr(j,[2 4 4]),'.-','LineWidth',gr.gr(j,6),'color',lineCol);
                                    end
                                end;
                                if(timestepMode)
                                    axis([0 T*timestep*1000 0 N]);
                                else
                                    axis([0 T 0 N]);
                                end
                                hold off
                                title(['group id:' num2str(length(groups))]);
                                ylabel('Cell Index');
                                xlabel('Time [ms]')
                                drawnow;
                            end
%                             if(plotFigure && saveImages  && longest_path>=min_group_path_to_plot)

                            if(plotFigure && saveImages  && length( find(gr.gr(:,6) == 2 ) )>min_multi_fanin_syns_to_plot && longest_path>=min_group_path_to_plot)
                                saveas(fig,['../output/' experimentName '/' num2str(trainedNet) '_poly_i_' num2str(length(groups)) strrep(mat2str(ppre{i_post}(strong_pre_id_to_specific_postCell(anchors))), ';', '_') '.fig']);
    %                             saveas(fig,['../output/' experimentName '/' num2str(trainedNet) '_poly_i_' num2str(length(groups)) strrep(mat2str(ppre{i_post}(strong_pre(anchors))), ';', '_') '.png']);
                                set(gcf,'PaperPositionMode','auto')
                                print(['../output/' experimentName '/' num2str(trainedNet) '_poly_i_' num2str(length(groups)) strrep(mat2str(ppre{i_post}(strong_pre_id_to_specific_postCell(anchors))), ';', '_')],'-dpng','-r0')
                            end
                        end;
                    end
                end

%                 length(gr.firings)

                % Now, get a different combination of the anchor neurons
                k=anchor_width;
                while k>0 & anchors(k)==length(strong_pre_id_to_specific_postCell)-(anchor_width-k)
                    k=k-1;
                end;

                if k==0, break, end;    % exhausted all possibilities

                anchors(k)=anchors(k)+1;
                for j=k+1:anchor_width
                    anchors(j)=anchors(j-1)+1;
                end;

                pause(0); % to avoid feezing when no groups are found for long time

            end;
        end;
    end;

    if (trainedNet)
        save(['../output/' experimentName '/groups_trained.mat'],'groups');
        
    else
        save(['../output/' experimentName '/groups_untrained.mat'],'groups');
    end
    disp(length(groups))
    clear;
end
