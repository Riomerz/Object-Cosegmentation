function [max_pooled]= maxpooling(llc_codes)
sze = size(llc_codes);
if(sze(1) == 1)
    max_pooled = llc_codes;
else
    max_pooled = max(llc_codes);
end



